"""
STDP Training Pipeline Module
==============================

Owner: CS2 / SNN Expert
Phase: Week 2 (Days 8-14)

This module implements the three-phase hybrid STDP training strategy:
- Phase 1: Pure STDP on layer 1 (unsupervised feature learning)
- Phase 2: Hybrid training (STDP frozen, backprop on layer 2)
- Phase 3: Fine-tuning (backprop on all layers)

Key Features:
- Comprehensive STDP statistics tracking
- Learning rate annealing for multi-timescale STDP
- Automatic phase transitions
- Checkpointing with STDP state
- Visualization-ready logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import json
import time
import logging

from src.model import HybridSTDP_SNN
from src.stdp import STDPConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Phase 1: Pure STDP Training (Unsupervised)
# ============================================

def train_stdp_epoch(
    model: HybridSTDP_SNN,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    max_epochs: int = 20,
    learning_rate_scale: float = 1.0
) -> Dict[str, float]:
    """
    Train one epoch using pure STDP (no backpropagation, no labels)

    This implements unsupervised feature learning where layer 1 learns
    to detect temporal patterns in ECG signals based on spike timing.

    Args:
        model: HybridSTDP_SNN model
        train_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        max_epochs: Total STDP epochs (for alpha annealing)
        learning_rate_scale: Global STDP learning rate multiplier

    Returns:
        metrics: Dictionary with STDP statistics
    """
    model.train()
    model.set_stdp_mode(True)

    # Aggregate statistics
    total_ltp_fast = 0
    total_ltd_fast = 0
    total_ltp_slow = 0
    total_ltd_slow = 0
    total_weight_change = 0.0
    total_alpha = 0.0
    total_divergence = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"STDP Epoch {epoch+1}/{max_epochs}")

    for batch_idx, (data, target) in enumerate(pbar):
        # Move to device (ignore labels for unsupervised STDP)
        data = data.to(device)

        # Forward pass (collect intermediate spikes)
        with torch.no_grad():  # STDP doesn't use autograd
            spikes, membrane, intermediate = model(data)

        # Apply STDP weight update to layer 1
        stdp_stats = model.apply_stdp_to_layer1(
            learning_rate_scale=learning_rate_scale,
            epoch=epoch,
            max_epochs=max_epochs
        )

        # Accumulate statistics
        total_ltp_fast += stdp_stats.get('ltp_fast', 0)
        total_ltd_fast += stdp_stats.get('ltd_fast', 0)
        total_ltp_slow += stdp_stats.get('ltp_slow', 0)
        total_ltd_slow += stdp_stats.get('ltd_slow', 0)
        total_weight_change += stdp_stats.get('weight_divergence', 0)
        total_alpha += stdp_stats.get('alpha', 0)
        total_divergence += stdp_stats.get('weight_divergence', 0)
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'LTP': total_ltp_fast // num_batches,
            'LTD': total_ltd_fast // num_batches,
            'alpha': total_alpha / num_batches,
            'Δw': total_weight_change / num_batches
        })

    # Calculate epoch metrics
    metrics = {
        'epoch': epoch,
        'phase': 'STDP',
        'total_ltp_fast': total_ltp_fast,
        'total_ltd_fast': total_ltd_fast,
        'total_ltp_slow': total_ltp_slow,
        'total_ltd_slow': total_ltd_slow,
        'avg_ltp_fast': total_ltp_fast / num_batches,
        'avg_ltd_fast': total_ltd_fast / num_batches,
        'ltp_ltd_ratio': total_ltp_fast / max(total_ltd_fast, 1),
        'avg_weight_change': total_weight_change / num_batches,
        'avg_alpha': total_alpha / num_batches,
        'avg_divergence': total_divergence / num_batches
    }

    logger.info(
        f"STDP Epoch {epoch+1}: "
        f"LTP={metrics['avg_ltp_fast']:.0f}, "
        f"LTD={metrics['avg_ltd_fast']:.0f}, "
        f"Ratio={metrics['ltp_ltd_ratio']:.3f}, "
        f"Alpha={metrics['avg_alpha']:.3f}"
    )

    return metrics


# ============================================
# Phase 2: Hybrid Training (STDP + Backprop)
# ============================================

def train_hybrid_epoch(
    model: HybridSTDP_SNN,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train one epoch with hybrid approach:
    - Layer 1: Frozen (STDP weights preserved)
    - Layer 2: Backprop (supervised classification)

    Args:
        model: HybridSTDP_SNN model (layer 1 should be frozen)
        train_loader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (only layer 2 parameters)
        device: Device to train on
        epoch: Current epoch number

    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    model.set_stdp_mode(False)  # Disable STDP, use backprop

    # Ensure layer 1 is frozen
    if not model.layer1_frozen:
        logger.warning("Layer 1 not frozen! Freezing now...")
        model.freeze_layer1()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Hybrid Epoch {epoch+1}")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        spikes, membrane, _ = model(data, return_intermediate=False)

        # Sum spikes over time for classification
        output = spikes.sum(dim=0)  # [batch, classes]

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass (only layer 2 updated)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total if total > 0 else 0
        })

    metrics = {
        'epoch': epoch,
        'phase': 'hybrid',
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }

    logger.info(
        f"Hybrid Epoch {epoch+1}: "
        f"Loss={metrics['loss']:.4f}, "
        f"Acc={metrics['accuracy']:.2f}%"
    )

    return metrics


# ============================================
# Phase 3: Fine-Tuning (Full Backprop)
# ============================================

def train_finetune_epoch(
    model: HybridSTDP_SNN,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train one epoch with full backpropagation (all layers)

    This fine-tunes both STDP-learned features and classification layer.

    Args:
        model: HybridSTDP_SNN model (all layers unfrozen)
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer (all parameters)
        device: Device to train on
        epoch: Current epoch number

    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    model.set_stdp_mode(False)

    # Ensure layer 1 is unfrozen
    if model.layer1_frozen:
        logger.warning("Layer 1 still frozen! Unfreezing now...")
        model.unfreeze_layer1()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        spikes, membrane, _ = model(data, return_intermediate=False)

        # Sum spikes over time
        output = spikes.sum(dim=0)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total if total > 0 else 0
        })

    metrics = {
        'epoch': epoch,
        'phase': 'finetune',
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }

    logger.info(
        f"Finetune Epoch {epoch+1}: "
        f"Loss={metrics['loss']:.4f}, "
        f"Acc={metrics['accuracy']:.2f}%"
    )

    return metrics


# ============================================
# Validation Function (All Phases)
# ============================================

def validate(
    model: HybridSTDP_SNN,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "validation"
) -> Dict[str, float]:
    """
    Validate model (works for all phases)

    Args:
        model: HybridSTDP_SNN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        phase: Phase name for logging

    Returns:
        metrics: Validation metrics
    """
    model.eval()
    model.set_stdp_mode(False)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f"{phase.capitalize()} Val"):
            data, target = data.to(device), target.to(device)

            # Forward pass
            spikes, membrane, _ = model(data, return_intermediate=False)
            output = spikes.sum(dim=0)

            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    metrics = {
        'val_loss': total_loss / len(val_loader),
        'val_accuracy': 100. * correct / total if total > 0 else 0
    }

    logger.info(
        f"{phase.capitalize()} Validation: "
        f"Loss={metrics['val_loss']:.4f}, "
        f"Acc={metrics['val_accuracy']:.2f}%"
    )

    return metrics


# ============================================
# Three-Phase Training Orchestrator
# ============================================

def train_three_phase(
    model: HybridSTDP_SNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    save_dir: str = 'models',
    # Phase 1 parameters
    stdp_epochs: int = 20,
    stdp_lr_scale: float = 1.0,
    # Phase 2 parameters
    hybrid_epochs: int = 30,
    hybrid_lr: float = 0.001,
    # Phase 3 parameters
    finetune_epochs: int = 20,
    finetune_lr: float = 0.0001,
    # General parameters
    log_interval: int = 5
) -> Dict:
    """
    Complete three-phase STDP training pipeline

    Phase 1 (Epochs 1-20): Pure STDP on layer 1 (unsupervised)
    Phase 2 (Epochs 21-50): Freeze layer 1, backprop on layer 2 (supervised)
    Phase 3 (Epochs 51-70): Unfreeze all, full backprop (fine-tuning)

    Args:
        model: HybridSTDP_SNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        save_dir: Directory to save models and logs
        stdp_epochs: Number of STDP-only epochs
        stdp_lr_scale: STDP learning rate scale
        hybrid_epochs: Number of hybrid epochs
        hybrid_lr: Hybrid phase learning rate
        finetune_epochs: Number of fine-tuning epochs
        finetune_lr: Fine-tuning learning rate
        log_interval: Logging frequency

    Returns:
        history: Complete training history with all metrics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Loss function (for phases 2 and 3)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'phase1_stdp': [],
        'phase2_hybrid': [],
        'phase3_finetune': [],
        'val_phase1': [],
        'val_phase2': [],
        'val_phase3': []
    }

    best_val_acc = 0.0
    total_epochs = stdp_epochs + hybrid_epochs + finetune_epochs

    logger.info("=" * 80)
    logger.info("🧠 THREE-PHASE STDP TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Phase 1 (STDP): {stdp_epochs} epochs")
    logger.info(f"Phase 2 (Hybrid): {hybrid_epochs} epochs")
    logger.info(f"Phase 3 (Finetune): {finetune_epochs} epochs")
    logger.info(f"Total: {total_epochs} epochs")
    logger.info("=" * 80)

    # ========================================
    # PHASE 1: Pure STDP Training
    # ========================================
    logger.info("\n🔬 PHASE 1: STDP-Only Training (Unsupervised Feature Learning)")
    logger.info("-" * 80)

    for epoch in range(stdp_epochs):
        stdp_metrics = train_stdp_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            max_epochs=stdp_epochs,
            learning_rate_scale=stdp_lr_scale
        )

        history['phase1_stdp'].append(stdp_metrics)

        # Periodic validation (just for monitoring, not used for STDP)
        if (epoch + 1) % log_interval == 0 or epoch == stdp_epochs - 1:
            val_metrics = validate(
                model, val_loader, criterion, device, phase="phase1"
            )
            history['val_phase1'].append({
                'epoch': epoch,
                **val_metrics
            })

            # Save STDP checkpoint
            checkpoint_path = Path(save_dir) / f'stdp_epoch_{epoch+1}.pt'
            save_stdp_checkpoint(
                model, None, epoch, stdp_metrics, checkpoint_path
            )

    logger.info(f"\n✅ Phase 1 Complete: STDP feature learning finished")
    logger.info(f"   Final Alpha: {history['phase1_stdp'][-1]['avg_alpha']:.3f}")
    logger.info(f"   Final LTP/LTD Ratio: {history['phase1_stdp'][-1]['ltp_ltd_ratio']:.3f}")

    # ========================================
    # PHASE 2: Hybrid Training
    # ========================================
    logger.info("\n🎯 PHASE 2: Hybrid Training (STDP frozen, Backprop on layer 2)")
    logger.info("-" * 80)

    # Freeze layer 1
    model.freeze_layer1()

    # Optimizer for layer 2 only
    hybrid_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hybrid_lr
    )

    for epoch in range(hybrid_epochs):
        global_epoch = stdp_epochs + epoch

        hybrid_metrics = train_hybrid_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=hybrid_optimizer,
            device=device,
            epoch=global_epoch
        )

        history['phase2_hybrid'].append(hybrid_metrics)

        # Validation
        val_metrics = validate(
            model, val_loader, criterion, device, phase="phase2"
        )
        history['val_phase2'].append({
            'epoch': global_epoch,
            **val_metrics
        })

        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            checkpoint_path = Path(save_dir) / 'best_hybrid_model.pt'
            save_stdp_checkpoint(
                model, hybrid_optimizer, global_epoch,
                {**hybrid_metrics, **val_metrics},
                checkpoint_path
            )
            logger.info(f"💾 Saved best hybrid model (Val Acc: {best_val_acc:.2f}%)")

    logger.info(f"\n✅ Phase 2 Complete: Best validation accuracy: {best_val_acc:.2f}%")

    # ========================================
    # PHASE 3: Fine-Tuning
    # ========================================
    logger.info("\n🔧 PHASE 3: Fine-Tuning (Full backprop on all layers)")
    logger.info("-" * 80)

    # Unfreeze layer 1
    model.unfreeze_layer1()

    # New optimizer for all parameters with lower learning rate
    finetune_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=finetune_lr
    )

    for epoch in range(finetune_epochs):
        global_epoch = stdp_epochs + hybrid_epochs + epoch

        finetune_metrics = train_finetune_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=finetune_optimizer,
            device=device,
            epoch=global_epoch
        )

        history['phase3_finetune'].append(finetune_metrics)

        # Validation
        val_metrics = validate(
            model, val_loader, criterion, device, phase="phase3"
        )
        history['val_phase3'].append({
            'epoch': global_epoch,
            **val_metrics
        })

        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            checkpoint_path = Path(save_dir) / 'best_finetuned_model.pt'
            save_stdp_checkpoint(
                model, finetune_optimizer, global_epoch,
                {**finetune_metrics, **val_metrics},
                checkpoint_path
            )
            logger.info(f"💾 Saved best finetuned model (Val Acc: {best_val_acc:.2f}%)")

    logger.info(f"\n✅ Phase 3 Complete: Final validation accuracy: {best_val_acc:.2f}%")

    # ========================================
    # Save Final History
    # ========================================
    history_path = Path(save_dir) / 'stdp_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n💾 Training history saved to {history_path}")
    logger.info("=" * 80)
    logger.info("🎉 THREE-PHASE TRAINING COMPLETE!")
    logger.info(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Total Epochs: {total_epochs}")
    logger.info("=" * 80)

    return history


# ============================================
# Checkpoint Management
# ============================================

def save_stdp_checkpoint(
    model: HybridSTDP_SNN,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict,
    path: str
):
    """
    Save STDP training checkpoint with full state

    Args:
        model: HybridSTDP_SNN model
        optimizer: Optimizer (can be None for STDP-only phases)
        epoch: Current epoch
        metrics: Training metrics
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': model.config.__dict__ if model.config is not None else None,
        'stdp_statistics': model.get_stdp_statistics()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_stdp_checkpoint(
    model: HybridSTDP_SNN,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cuda'
) -> Dict:
    """
    Load STDP training checkpoint

    Args:
        model: HybridSTDP_SNN model
        optimizer: Optimizer (optional)
        path: Checkpoint path
        device: Device to load on

    Returns:
        checkpoint: Full checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint and 'val_accuracy' in checkpoint['metrics']:
        logger.info(f"   Validation Accuracy: {checkpoint['metrics']['val_accuracy']:.2f}%")

    return checkpoint


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 STDP Training Module Loaded Successfully")
    print("\nTo use:")
    print("   from src.train_stdp import train_three_phase")
    print("   from src.model import HybridSTDP_SNN")
    print("   from src.stdp import STDPConfig")
    print()
    print("   config = STDPConfig(use_homeostasis=True, use_multiscale=True)")
    print("   model = HybridSTDP_SNN(config=config)")
    print("   history = train_three_phase(model, train_loader, val_loader)")
