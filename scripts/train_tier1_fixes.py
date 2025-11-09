"""
Comprehensive Training Script for Tier 1 Fixes
==============================================

Trains improved models with:
- Fix #2: FocalLoss for class-weighted training
- Fix #3: DeepSNN architecture for better capacity
- Clinical metrics tracking
- Sensitivity-based early stopping

Usage:
    # Train SimpleSNN with FocalLoss (Fix #2 only)
    python scripts/train_tier1_fixes.py --model simple --epochs 30

    # Train DeepSNN with FocalLoss (Fix #2 + #3)
    python scripts/train_tier1_fixes.py --model deep --epochs 30

    # Train both and compare
    python scripts/train_tier1_fixes.py --model both --epochs 30
"""

import torch
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SimpleSNN, DeepSNN, WiderSNN
from src.train import train_model
from src.losses import FocalLoss, get_loss_function
from src.data import ECGDataset
from torch.utils.data import DataLoader


def load_data(batch_size=32):
    """Load training and validation data"""
    print("\n📦 Loading datasets...")

    # Load data
    train_data = torch.load('data/synthetic/train_data.pt')
    val_data = torch.load('data/synthetic/val_data.pt')

    print(f"✓ Train: {len(train_data['signals'])} samples")
    print(f"✓ Val:   {len(val_data['signals'])} samples")

    # Create datasets
    train_dataset = ECGDataset(
        signals=train_data['signals'],
        labels=train_data['labels'],
        num_steps=100,
        gain=10.0
    )

    val_dataset = ECGDataset(
        signals=val_data['signals'],
        labels=val_data['labels'],
        num_steps=100,
        gain=10.0
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for performance
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def train_single_model(
    model_type='deep',
    num_epochs=30,
    learning_rate=0.001,
    batch_size=32,
    alpha=0.75,
    gamma=2.0,
    weight_decay=1e-4,
    device='cuda'
):
    """
    Train a single model configuration

    Args:
        model_type: 'simple', 'wider', or 'deep'
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        alpha: FocalLoss alpha parameter (weight for arrhythmia class)
        gamma: FocalLoss gamma parameter (focusing parameter)
        weight_decay: L2 regularization weight
        device: Device for training

    Returns:
        training_history, best_sensitivity, model_path
    """
    print("\n" + "=" * 80)
    print(f"TRAINING: {model_type.upper()}SNN with FocalLoss")
    print("=" * 80)

    # Create model
    if model_type == 'simple':
        model = SimpleSNN()
        save_name = f'simple_focal_model.pt'
        print("Architecture: SimpleSNN (2500→128→2, ~320K params)")
    elif model_type == 'wider':
        model = WiderSNN(hidden_size=256, dropout=0.2)
        save_name = f'wider_focal_model.pt'
        print("Architecture: WiderSNN (2500→256→2, ~640K params)")
    elif model_type == 'deep':
        model = DeepSNN(hidden_sizes=[256, 128], dropout=0.3)
        save_name = f'deep_focal_model.pt'
        print("Architecture: DeepSNN (2500→256→128→2, ~673K params)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load data
    train_loader, val_loader = load_data(batch_size=batch_size)

    # Create FocalLoss criterion
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    print(f"\nLoss Function: FocalLoss(alpha={alpha}, gamma={gamma})")
    print(f"Regularization: L2 weight_decay={weight_decay}")

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir='models',
        criterion=criterion,
        weight_decay=weight_decay,
        sensitivity_target=0.95,
        early_stopping_patience=10,
        save_name=save_name
    )

    # Extract best sensitivity
    best_epoch = len(history['val_sensitivity']) - 1
    for i, sens in enumerate(history['val_sensitivity']):
        if sens == max(history['val_sensitivity']):
            best_epoch = i
            break

    best_sensitivity = history['val_sensitivity'][best_epoch]
    best_specificity = history['val_specificity'][best_epoch]
    best_accuracy = history['val_acc'][best_epoch]

    print(f"\n✅ Training complete for {model_type.upper()}SNN")
    print(f"   Best Epoch: {best_epoch + 1}")
    print(f"   Sensitivity: {best_sensitivity:.1f}%")
    print(f"   Specificity: {best_specificity:.1f}%")
    print(f"   Accuracy: {best_accuracy:.1f}%")
    print(f"   Model saved: models/{save_name}")

    return history, best_sensitivity, f'models/{save_name}'


def compare_models(results):
    """Compare training results from multiple models"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<15} {'Sensitivity':<15} {'Specificity':<15} {'Accuracy':<15} {'Targets Met'}")
    print("-" * 80)

    for model_name, history, best_sens, model_path in results:
        # Find best epoch
        best_epoch = history['val_sensitivity'].index(max(history['val_sensitivity']))
        sens = history['val_sensitivity'][best_epoch]
        spec = history['val_specificity'][best_epoch]
        acc = history['val_acc'][best_epoch]

        targets_met = "✅" if sens >= 95 and spec >= 85 else "❌"

        print(f"{model_name:<15} {sens:>7.1f}%{' '*7} {spec:>7.1f}%{' '*7} {acc:>7.1f}%{' '*7} {targets_met}")

    print("\n" + "=" * 80)

    # Determine best model
    best_model = max(results, key=lambda x: x[1])
    print(f"\n🏆 BEST MODEL: {best_model[0]} (Sensitivity: {best_model[1]:.1f}%)")
    print(f"   Path: {best_model[2]}")

    return best_model


def main():
    parser = argparse.ArgumentParser(description='Train Tier 1 Fixes Models')
    parser.add_argument('--model', type=str, default='deep',
                       choices=['simple', 'wider', 'deep', 'both'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.75,
                       help='FocalLoss alpha parameter')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='FocalLoss gamma parameter')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 regularization weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for training (cuda/cpu)')
    args = parser.parse_args()

    print("=" * 80)
    print("TIER 1 FIXES TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model(s):      {args.model}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  FocalLoss:     alpha={args.alpha}, gamma={args.gamma}")
    print(f"  Weight Decay:  {args.weight_decay}")
    print(f"  Device:        {args.device}")

    # Ensure CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'

    results = []

    # Train models
    if args.model == 'both':
        # Train both SimpleSNN and DeepSNN
        for model_type in ['simple', 'deep']:
            history, best_sens, model_path = train_single_model(
                model_type=model_type,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                alpha=args.alpha,
                gamma=args.gamma,
                weight_decay=args.weight_decay,
                device=args.device
            )
            results.append((f"{model_type.upper()}SNN", history, best_sens, model_path))

        # Compare results
        best_model = compare_models(results)

    else:
        # Train single model
        history, best_sens, model_path = train_single_model(
            model_type=args.model,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            alpha=args.alpha,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            device=args.device
        )

        print(f"\n✅ Training complete!")
        print(f"   Model saved: {model_path}")
        print(f"   Best Sensitivity: {best_sens:.1f}%")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Run Phase 2 evaluation on trained model:")
    print(f"   python scripts/comprehensive_evaluation.py --model {model_path if args.model != 'both' else best_model[2]}")
    print("\n2. Compare with baseline (models/best_model.pt)")
    print("\n3. If targets met, update production model checkpoint")
    print("=" * 80)


if __name__ == '__main__':
    main()
