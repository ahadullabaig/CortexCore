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
        # SNN returns (spikes, membrane): [time_steps, batch, classes]
        spikes, membrane = model(data)

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
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model

    TODO:
        - Add more metrics (precision, recall, F1)
        - Implement confusion matrix
        - Add clinical validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            # Forward pass
            # SNN returns (spikes, membrane): [time_steps, batch, classes]
            spikes, membrane = model(data)

            # Sum spikes over time dimension for classification
            output = spikes.sum(dim=0)  # [batch, classes]

            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }

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
    log_interval: int = 10
) -> Dict:
    """
    Full training pipeline

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save models
        log_interval: Logging frequency

    Returns:
        Training history

    TODO:
        - Add early stopping
        - Implement learning rate scheduling
        - Add wandb/mlflow logging
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # TODO: Choose appropriate criterion for SNN
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"🏋️  Training on {device}")
    print(f"📊 Epochs: {num_epochs}, Learning rate: {learning_rate}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = Path(save_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
            }, checkpoint_path)
            print(f"💾 Saved best model (Val Acc: {val_metrics['accuracy']:.2f}%)")

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
