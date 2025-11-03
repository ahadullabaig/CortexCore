"""
Complete SNN Training Script for MVP
=====================================

This script trains the SimpleSNN model on generated ECG data.
Target: ≥85% validation accuracy
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.model import SimpleSNN, measure_energy_efficiency
from src.data import ECGDataset
from src.train import train_epoch, validate
from src.utils import set_seed, get_device
import json

def main():
    # Configuration
    CONFIG = {
        'data_dir': 'data/synthetic',
        'model_dir': 'models',
        'results_dir': 'results/metrics',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'hidden_size': 128,
        'num_steps': 100,  # Time steps for SNN
        'beta': 0.9,  # Membrane decay
        'seed': 42
    }

    print("="*60)
    print("🧠 SimpleSNN Training - MVP")
    print("="*60)

    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    device = get_device()

    print(f"\n📝 Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print(f"   Epochs: {CONFIG['num_epochs']}")
    print(f"   Time steps: {CONFIG['num_steps']}")

    # Load data
    print(f"\n📂 Loading datasets...")
    train_data = torch.load(f"{CONFIG['data_dir']}/train_data.pt")
    val_data = torch.load(f"{CONFIG['data_dir']}/val_data.pt")

    print(f"   Train: {train_data['signals'].shape[0]} samples")
    print(f"   Val: {val_data['signals'].shape[0]} samples")

    # Create datasets
    train_dataset = ECGDataset(
        signals=train_data['signals'].numpy(),
        labels=train_data['labels'].numpy(),
        encode_spikes=True,
        num_steps=CONFIG['num_steps']
    )

    val_dataset = ECGDataset(
        signals=val_data['signals'].numpy(),
        labels=val_data['labels'].numpy(),
        encode_spikes=True,
        num_steps=CONFIG['num_steps']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    print(f"\n🏗️  Creating SimpleSNN model...")
    input_size = train_data['signals'].shape[1]  # Signal length
    model = SimpleSNN(
        input_size=input_size,
        hidden_size=CONFIG['hidden_size'],
        output_size=2,
        beta=CONFIG['beta']
    )

    model = model.to(device)
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {CONFIG['hidden_size']}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Training loop
    print(f"\n🏋️  Starting training...")
    print("="*60)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    Path(CONFIG['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n📅 Epoch {epoch + 1}/{CONFIG['num_epochs']}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Print metrics
        print(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = Path(CONFIG['model_dir']) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'config': CONFIG
            }, checkpoint_path)
            print(f"   💾 Saved best model (Val Acc: {val_metrics['accuracy']:.2f}%)")

        # Check if MVP target achieved
        if val_metrics['accuracy'] >= 85.0:
            print(f"\n🎉 MVP TARGET ACHIEVED! Val Acc: {val_metrics['accuracy']:.2f}% ≥ 85%")

    # Final results
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"MVP Target (≥85%): {'✅ ACHIEVED' if best_val_acc >= 85 else '❌ NOT MET'}")

    # Save training history
    history_path = Path(CONFIG['results_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved to: {history_path}")

    # Measure energy efficiency
    print(f"\n⚡ Measuring energy efficiency...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        data, _ = sample_batch
        data = data.to(device)
        spikes, _ = model(data)

        energy_metrics = measure_energy_efficiency(spikes)
        print(f"   Spike rate: {energy_metrics['spike_rate']:.4f}")
        print(f"   Sparsity: {energy_metrics['sparsity']:.2f}%")
        print(f"   Total spikes: {energy_metrics['total_spikes']:.0f}")

    print(f"\n🎓 Next steps:")
    print(f"   1. Run inference: python src/inference.py")
    print(f"   2. Start demo: bash scripts/04_run_demo.sh")
    print(f"   3. Run tests: bash scripts/05_test_integration.sh")

if __name__ == "__main__":
    main()
