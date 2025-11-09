"""
Training Pipeline Module
========================

Owner: CS2 / SNN Expert (with CS1 infrastructure support)

Responsibilities:
- Training loops
- Validation
- Checkpoint management
- Logging and monitoring

Phase: Days 3-30
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import json
import time

# ============================================
# TODO: Day 3-4 - Basic Training Loop
# ============================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch

    Returns:
        Dictionary with training metrics

    TODO:
        - Add gradient clipping
        - Implement mixed precision training
        - Add learning rate scheduling
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        # SNN returns (spikes, membrane) or (spikes, membrane, intermediate)
        model_output = model(data)

        # Handle both 2-tuple and 3-tuple returns
        if isinstance(model_output, tuple):
            if len(model_output) == 3:
                spikes, membrane, intermediate = model_output
            else:
                spikes, membrane = model_output
        else:
            raise ValueError("Model output must be a tuple")

        # Sum spikes over time dimension for classification
        output = spikes.sum(dim=0)  # [batch, classes]

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total if total > 0 else 0
        })

    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_clinical_metrics: bool = True
) -> Dict[str, float]:
    """
    Validate model with clinical metrics

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device for computation
        return_clinical_metrics: If True, calculates sensitivity, specificity, etc.

    Returns:
        Dictionary with loss, accuracy, and optionally clinical metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # For clinical metrics
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            # Forward pass
            # SNN returns (spikes, membrane) or (spikes, membrane, intermediate)
            model_output = model(data)

            # Handle both 2-tuple and 3-tuple returns
            if isinstance(model_output, tuple):
                if len(model_output) == 3:
                    spikes, membrane, intermediate = model_output
                else:
                    spikes, membrane = model_output
            else:
                raise ValueError("Model output must be a tuple")

            # Sum spikes over time dimension for classification
            output = spikes.sum(dim=0)  # [batch, classes]

            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Store for clinical metrics
            if return_clinical_metrics:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }

    # Calculate clinical metrics if requested
    if return_clinical_metrics and len(all_targets) > 0:
        import numpy as np
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)

        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        metrics.update({
            'sensitivity': sensitivity * 100,  # Convert to percentage
            'specificity': specificity * 100,
            'precision': precision * 100,
            'npv': npv * 100,
            'f1_score': f1 * 100,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })

    return metrics


# ============================================
# TODO: Day 4-7 - Full Training Pipeline
# ============================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    save_dir: str = 'models',
    log_interval: int = 10,
    criterion: Optional[nn.Module] = None,
    weight_decay: float = 0.0,
    sensitivity_target: float = 0.95,
    early_stopping_patience: int = 10,
    save_name: str = 'best_model.pt'
) -> Dict:
    """
    Full training pipeline with clinical metrics and early stopping

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save models
        log_interval: Logging frequency
        criterion: Loss function (if None, uses CrossEntropyLoss)
        weight_decay: L2 regularization weight (default: 0.0)
        sensitivity_target: Target sensitivity for early stopping (default: 0.95)
        early_stopping_patience: Epochs to wait before stopping (default: 10)
        save_name: Name for saved model checkpoint (default: 'best_model.pt')

    Returns:
        Training history with clinical metrics

    Clinical Early Stopping:
        Saves best model when sensitivity >= sensitivity_target AND specificity >= 85%
        Falls back to highest sensitivity if target not met
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use provided criterion or default to CrossEntropyLoss
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Move criterion to device if it has weights (e.g., WeightedCE)
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training history with clinical metrics
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

    best_sensitivity = 0.0
    best_g_mean = 0.0  # Geometric mean of sensitivity and specificity
    best_val_acc = 0.0
    epochs_no_improve = 0
    target_met = False
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"🏋️  Training on {device}")
    print(f"📊 Epochs: {num_epochs}, Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"🎯 Target: Sensitivity ≥ {sensitivity_target*100:.0f}%, Specificity ≥ 85%")
    print(f"📉 Loss function: {criterion.__class__.__name__}")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate with clinical metrics
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

        # Check if clinical targets met
        sensitivity = val_metrics.get('sensitivity', 0) / 100.0
        specificity = val_metrics.get('specificity', 0) / 100.0
        targets_met = sensitivity >= sensitivity_target and specificity >= 0.85

        # Save best model based on clinical criteria
        save_model = False
        g_mean = (sensitivity * specificity) ** 0.5  # Geometric mean balances both metrics

        if targets_met and not target_met:
            # First time meeting targets
            target_met = True
            save_model = True
            print(f"🎯 Clinical targets MET! Sensitivity: {sensitivity*100:.1f}%, Specificity: {specificity*100:.1f}%")
        elif target_met and targets_met:
            # Targets already met, save if sensitivity improved
            if sensitivity > best_sensitivity:
                save_model = True
                print(f"✓ Improved sensitivity: {best_sensitivity*100:.1f}% → {sensitivity*100:.1f}%")
        elif not target_met:
            # Targets not yet met, save best BALANCED performance (G-mean)
            # This favors balanced models over extreme sensitivity/specificity
            if g_mean > best_g_mean:
                save_model = True
                print(f"✓ Best G-mean: {g_mean*100:.1f}% (Sens: {sensitivity*100:.1f}%, Spec: {specificity*100:.1f}%)")

        if save_model:
            best_sensitivity = sensitivity
            best_g_mean = g_mean
            best_val_acc = val_metrics['accuracy']
            epochs_no_improve = 0

            checkpoint_path = Path(save_dir) / save_name
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_sensitivity': sensitivity * 100,
                'val_specificity': specificity * 100,
                'history': history,
                'targets_met': targets_met
            }

            # Add model config if available
            if hasattr(model, 'config'):
                checkpoint['config'] = model.config

            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved best model to {save_name}")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⏹️  Early stopping: No improvement for {early_stopping_patience} epochs")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Sensitivity: {best_sensitivity*100:.1f}%")
    if target_met:
        print(f"✅ Clinical targets achieved!")
    else:
        print(f"⚠️  Clinical targets not fully met (Sensitivity: {best_sensitivity*100:.1f}% < {sensitivity_target*100:.0f}%)")
    print("=" * 70)

    return history


# ============================================
# TODO: Day 8+ - Advanced Training Features
# ============================================

def train_with_stdp():
    """TODO: Implement STDP training"""
    raise NotImplementedError("Week 2 task")


def train_hybrid_model():
    """TODO: Implement hybrid SNN-ANN training"""
    raise NotImplementedError("Week 2 task")


def hyperparameter_tuning():
    """TODO: Implement with Optuna"""
    raise NotImplementedError("Week 3 task")


# ============================================
# Checkpoint Management
# ============================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str
):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cuda'
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 Testing training module...")
    print("✅ Training module loaded successfully")
    print("\n📝 To use:")
    print("   1. Import: from src.train import train_model")
    print("   2. Create model and dataloaders")
    print("   3. Call: train_model(model, train_loader, val_loader)")
