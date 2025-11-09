#!/usr/bin/env python3
"""
MIT-BIH Transfer Learning Training Script
==========================================

Two-stage transfer learning strategy:
    Stage 1 (Freeze): Freeze layer 1, train layer 2-3 for 20 epochs @ lr=0.0001
    Stage 2 (Fine-tune): Unfreeze all, train for 30 epochs @ lr=0.00005

Usage:
    # Stage 1 (required first)
    python scripts/train_mitbih_transfer.py --stage 1 --num_epochs 20

    # Stage 2 (run after Stage 1)
    python scripts/train_mitbih_transfer.py --stage 2 --num_epochs 30 \
        --pretrained_model models/mitbih_stage1/best_model.pt

    # Full pipeline (both stages)
    python scripts/train_mitbih_transfer.py --stage both

Arguments:
    --stage: Training stage (1, 2, or both)
    --pretrained_model: Path to pretrained model (default: models/deep_focal_model.pt)
    --data_dir: Path to MIT-BIH processed data (default: data/mitbih_processed)
    --output_dir: Output directory for checkpoints (default: models/mitbih_transfer)
    --batch_size: Batch size (default: 32)
    --num_epochs: Number of epochs per stage (default: 20 for stage 1, 30 for stage 2)
    --learning_rate: Learning rate (default: 0.0001 for stage 1, 0.00005 for stage 2)
    --weight_decay: L2 regularization (default: 0.001)
    --dropout: Dropout probability (default: 0.5 for regularization)
    --focal_alpha: FocalLoss alpha (default: 0.75)
    --focal_gamma: FocalLoss gamma (default: 2.0)
    --early_stopping_patience: Patience for early stopping (default: 10)
    --device: Device (cuda/cpu/mps, default: auto-detect)
    --seed: Random seed (default: 42)
    --augment: Enable data augmentation (flag)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import DeepSNN
from src.losses import FocalLoss
from src.train import train_epoch, validate
from src.data import load_dataset
from src.utils import set_seed, get_device


def load_pretrained_model(
    model_path: str,
    device: torch.device,
    dropout: float = 0.5
) -> Tuple[DeepSNN, Dict]:
    """
    Load pretrained model from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: Device to load model to
        dropout: Dropout probability (may differ from pretrained)

    Returns:
        model: Loaded DeepSNN model
        checkpoint: Checkpoint dictionary with metadata
    """
    print(f"\n📥 Loading pretrained model from {model_path}")

    # PyTorch 2.6+ compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(model_path, map_location=device)

    # Check if checkpoint contains config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   Model architecture: {config.get('architecture', 'Unknown')}")
        print(f"   Hidden sizes: {config.get('hidden_sizes', 'Unknown')}")
    else:
        print("   Warning: No config in checkpoint, using default DeepSNN")

    # Create model with possibly different dropout for transfer learning
    model = DeepSNN(dropout=dropout)
    model.to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded model state dict")
    else:
        # Checkpoint may be raw state_dict (backward compatibility)
        model.load_state_dict(checkpoint)
        print(f"   Loaded raw state dict")

    # Print checkpoint metrics if available
    if 'val_acc' in checkpoint:
        print(f"   Pretrained val accuracy: {checkpoint['val_acc']:.2f}%")
    if 'val_sensitivity' in checkpoint:
        print(f"   Pretrained sensitivity: {checkpoint['val_sensitivity']:.2f}%")
    if 'val_specificity' in checkpoint:
        print(f"   Pretrained specificity: {checkpoint['val_specificity']:.2f}%")

    return model, checkpoint


def freeze_layer(model: DeepSNN, layer_num: int = 1):
    """
    Freeze specified layer (prevent gradient updates)

    Args:
        model: DeepSNN model
        layer_num: Layer to freeze (1, 2, or 3)
    """
    if layer_num == 1:
        for param in model.fc1.parameters():
            param.requires_grad = False
        print(f"   ❄️  Layer 1 (fc1) frozen: {sum(p.numel() for p in model.fc1.parameters())} params")

    elif layer_num == 2:
        for param in model.fc2.parameters():
            param.requires_grad = False
        print(f"   ❄️  Layer 2 (fc2) frozen: {sum(p.numel() for p in model.fc2.parameters())} params")

    elif layer_num == 3:
        for param in model.fc3.parameters():
            param.requires_grad = False
        print(f"   ❄️  Layer 3 (fc3) frozen: {sum(p.numel() for p in model.fc3.parameters())} params")


def unfreeze_all_layers(model: DeepSNN):
    """
    Unfreeze all layers (enable gradient updates)

    Args:
        model: DeepSNN model
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔓 All layers unfrozen: {trainable_params:,} trainable params")


def train_stage(
    model: DeepSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
    early_stopping_patience: int = 10,
    sensitivity_target: float = 0.95
) -> Dict:
    """
    Train single stage with early stopping

    Args:
        model: DeepSNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        stage: Stage number (1 or 2)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization
        criterion: Loss function
        device: Device
        output_dir: Output directory for checkpoints
        early_stopping_patience: Patience for early stopping
        sensitivity_target: Target sensitivity for early stopping

    Returns:
        Training history
    """
    print(f"\n{'='*80}")
    print(f"STAGE {stage}: {'FREEZE LAYER 1' if stage == 1 else 'FULL FINE-TUNING'}")
    print(f"{'='*80}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"🎯 Target: Sensitivity ≥ {sensitivity_target*100:.0f}%, Specificity ≥ 85%")
    print(f"📉 Loss: {criterion}")
    print(f"⚙️  Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})")
    print(f"⏱️  Epochs: {num_epochs}, Patience: {early_stopping_patience}")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': [],
        'val_f1_score': []
    }

    # Early stopping tracking
    best_g_mean = 0.0
    best_sensitivity = 0.0
    epochs_no_improve = 0
    target_met = False

    # Create stage output directory
    stage_dir = output_dir / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, return_clinical_metrics=True)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics.get('sensitivity', 0))
        history['val_specificity'].append(val_metrics.get('specificity', 0))
        history['val_precision'].append(val_metrics.get('precision', 0))
        history['val_f1_score'].append(val_metrics.get('f1_score', 0))

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"        Sensitivity: {val_metrics.get('sensitivity', 0):.1f}%, "
              f"Specificity: {val_metrics.get('specificity', 0):.1f}%, "
              f"F1: {val_metrics.get('f1_score', 0):.1f}%")
        print(f"        FN: {val_metrics.get('false_negatives', 0)}, "
              f"FP: {val_metrics.get('false_positives', 0)}")

        # Early stopping logic (G-mean)
        sensitivity = val_metrics.get('sensitivity', 0) / 100.0
        specificity = val_metrics.get('specificity', 0) / 100.0
        g_mean = (sensitivity * specificity) ** 0.5
        targets_met = sensitivity >= sensitivity_target and specificity >= 0.85

        save_model = False

        if targets_met and not target_met:
            target_met = True
            save_model = True
            print(f"🎯 Clinical targets MET! G-mean: {g_mean*100:.1f}%")
        elif target_met and targets_met:
            if sensitivity > best_sensitivity:
                save_model = True
                print(f"✓ Improved sensitivity: {best_sensitivity*100:.1f}% → {sensitivity*100:.1f}%")
        elif not target_met:
            if g_mean > best_g_mean:
                save_model = True
                print(f"✓ Best G-mean: {g_mean*100:.1f}% (Sens: {sensitivity*100:.1f}%, Spec: {specificity*100:.1f}%)")

        if save_model:
            best_g_mean = g_mean
            best_sensitivity = sensitivity
            epochs_no_improve = 0

            # Save checkpoint
            checkpoint_path = stage_dir / 'best_model.pt'
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_sensitivity': sensitivity * 100,
                'val_specificity': specificity * 100,
                'val_g_mean': g_mean * 100,
                'history': history,
                'targets_met': targets_met,
                'config': model.config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved best model to {checkpoint_path.relative_to(project_root)}")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⏹️  Early stopping: No improvement for {early_stopping_patience} epochs")
            break

    # Training summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"STAGE {stage} COMPLETE")
    print(f"{'='*80}")
    print(f"Training time: {elapsed_time/60:.1f} minutes")
    print(f"Best G-mean: {best_g_mean*100:.1f}%")
    print(f"Best Sensitivity: {best_sensitivity*100:.1f}%")
    if target_met:
        print(f"✅ Clinical targets achieved!")
    else:
        print(f"⚠️  Clinical targets not fully met")
    print(f"Best model saved to: {stage_dir / 'best_model.pt'}")

    # Save history
    history_path = stage_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path.relative_to(project_root)}")

    return history


def main():
    parser = argparse.ArgumentParser(description='MIT-BIH Transfer Learning Training')

    # Training stage
    parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'],
                        help='Training stage (1=freeze layer 1, 2=full fine-tune, both=run both)')

    # Model and data paths
    parser.add_argument('--pretrained_model', type=str, default='models/deep_focal_model.pt',
                        help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str, default='data/mitbih_processed',
                        help='Path to MIT-BIH processed data')
    parser.add_argument('--output_dir', type=str, default='models/mitbih_transfer',
                        help='Output directory for checkpoints')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs per stage (default: 20 for stage 1, 30 for stage 2)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: 0.0001 for stage 1, 0.00005 for stage 2)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='L2 regularization weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (high for small dataset)')

    # Loss function
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='FocalLoss alpha (class weighting)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='FocalLoss gamma (focusing parameter)')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--sensitivity_target', type=float, default=0.95,
                        help='Target sensitivity for early stopping')

    # System
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps, default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (not implemented yet)')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    # Print configuration
    print("="*80)
    print("MIT-BIH TRANSFER LEARNING TRAINING")
    print("="*80)
    print(f"Pretrained model: {args.pretrained_model}")
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device:           {device}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Dropout:          {args.dropout}")
    print(f"FocalLoss:        α={args.focal_alpha}, γ={args.focal_gamma}")
    print(f"Early stopping:   patience={args.early_stopping_patience}")
    print(f"Random seed:      {args.seed}")
    print("="*80)

    # Load datasets
    print("\n📂 Loading MIT-BIH datasets...")
    data_dir = Path(args.data_dir)

    train_loader = load_dataset(
        str(data_dir / 'train_ecg.pt'),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = load_dataset(
        str(data_dir / 'val_ecg.pt'),
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = load_dataset(
        str(data_dir / 'test_ecg.pt'),
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val:   {len(val_loader.dataset)} samples")
    print(f"   Test:  {len(test_loader.dataset)} samples")

    # Create loss function
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Configuration saved to: {config_path}")

    # Execute training stages
    if args.stage == '1' or args.stage == 'both':
        # STAGE 1: Freeze layer 1, train layer 2-3
        num_epochs = args.num_epochs if args.num_epochs is not None else 20
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.0001

        # Load pretrained model
        model, _ = load_pretrained_model(args.pretrained_model, device, dropout=args.dropout)

        # Freeze layer 1
        freeze_layer(model, layer_num=1)

        # Train Stage 1
        history_stage1 = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            stage=1,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            criterion=criterion,
            device=device,
            output_dir=output_dir,
            early_stopping_patience=args.early_stopping_patience,
            sensitivity_target=args.sensitivity_target
        )

        # Update pretrained model path for Stage 2
        if args.stage == 'both':
            args.pretrained_model = str(output_dir / 'stage1' / 'best_model.pt')

    if args.stage == '2' or args.stage == 'both':
        # STAGE 2: Unfreeze all, full fine-tuning
        num_epochs = args.num_epochs if args.num_epochs is not None else 30
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.00005

        # Load model (either from Stage 1 or provided pretrained)
        model, _ = load_pretrained_model(args.pretrained_model, device, dropout=args.dropout)

        # Unfreeze all layers
        unfreeze_all_layers(model)

        # Train Stage 2
        history_stage2 = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            stage=2,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            criterion=criterion,
            device=device,
            output_dir=output_dir,
            early_stopping_patience=args.early_stopping_patience,
            sensitivity_target=args.sensitivity_target
        )

    # Final test evaluation (if Stage 2 completed)
    if args.stage == '2' or args.stage == 'both':
        print(f"\n{'='*80}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*80}")

        # Load best Stage 2 model
        best_model_path = output_dir / 'stage2' / 'best_model.pt'
        model, checkpoint = load_pretrained_model(str(best_model_path), device, dropout=args.dropout)

        # Evaluate on test set
        test_metrics = validate(model, test_loader, criterion, device, return_clinical_metrics=True)

        print(f"\nTest Results:")
        print(f"  Accuracy:    {test_metrics['accuracy']:.2f}%")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.2f}%")
        print(f"  Specificity: {test_metrics['specificity']:.2f}%")
        print(f"  Precision:   {test_metrics['precision']:.2f}%")
        print(f"  F1 Score:    {test_metrics['f1_score']:.2f}%")
        print(f"  Loss:        {test_metrics['loss']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {test_metrics['true_positives']}, TN: {test_metrics['true_negatives']}")
        print(f"  FP: {test_metrics['false_positives']}, FN: {test_metrics['false_negatives']}")

        # Save test results
        test_results_path = output_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nTest results saved to: {test_results_path}")

        # Check if targets met
        if test_metrics['sensitivity'] >= args.sensitivity_target * 100 and test_metrics['specificity'] >= 85:
            print(f"\n✅ SUCCESS! Real data targets achieved!")
            print(f"   Sensitivity: {test_metrics['sensitivity']:.1f}% ≥ {args.sensitivity_target*100:.0f}%")
            print(f"   Specificity: {test_metrics['specificity']:.1f}% ≥ 85%")
        else:
            print(f"\n⚠️  Targets not fully met on test set")
            if test_metrics['sensitivity'] < args.sensitivity_target * 100:
                print(f"   Sensitivity: {test_metrics['sensitivity']:.1f}% < {args.sensitivity_target*100:.0f}% (target)")
            if test_metrics['specificity'] < 85:
                print(f"   Specificity: {test_metrics['specificity']:.1f}% < 85% (target)")

    print(f"\n{'='*80}")
    print("TRANSFER LEARNING COMPLETE!")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review training history: {output_dir}/stage*/training_history.json")
    print(f"  2. Review test results: {output_dir}/test_results.json")
    print(f"  3. If targets not met, consider:")
    print(f"     - Lower SQI threshold (rerun preprocessing)")
    print(f"     - Add data augmentation")
    print(f"     - Tune hyperparameters")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
