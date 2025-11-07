#!/bin/bash

# ============================================
# CortexCore - MODEL TRAINING
# ============================================
# Trains SNN model on synthetic data
# Usage: bash scripts/03_train_mvp_model.sh
# Time: ~5-15 minutes depending on GPU

set -e  # Exit on error

echo "🧠 ============================================"
echo "🧠 SNN MODEL TRAINING"
echo "🧠 ============================================"
echo ""

# ==========================================
# Detect Python (venv or system)
# ==========================================

if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo "🐍 Using venv Python: $PYTHON"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
    echo "🐍 Using activated venv Python"
else
    PYTHON="python3"
    echo "⚠️  WARNING: venv not found, using system Python"
    echo "   Run 'bash scripts/01_setup_environment.sh' first for best results"
fi
echo ""

# ==========================================
# Configuration
# ==========================================

# Load from .env if exists
if [ -f ".env" ]; then
    source .env
fi

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
NUM_EPOCHS=${NUM_EPOCHS:-50}
DEVICE=${DEVICE:-cuda}

echo "📋 Configuration:"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Epochs: $NUM_EPOCHS"
echo "   Device: $DEVICE"
echo ""

# ==========================================
# Check Data Availability
# ==========================================

echo "🔍 Checking for data..."

if [ ! -f "data/synthetic/mvp_dataset.pt" ]; then
    echo "   ❌ ERROR: MVP dataset not found"
    echo "   Please run: bash scripts/02_generate_mvp_data.sh"
    exit 1
fi

echo "   ✅ Dataset found"
echo ""

# ==========================================
# Check Dependencies
# ==========================================

echo "🔍 Checking dependencies..."

$PYTHON -c "import torch; import snntorch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All required packages available"
else
    echo "   ❌ ERROR: Missing dependencies"
    echo "   Please run: bash scripts/01_setup_environment.sh"
    exit 1
fi

echo ""

# ==========================================
# Train Model
# ==========================================

echo "🏋️  Training SNN model..."
echo "   This may take 5-15 minutes..."
echo ""

# Create training script using actual SimpleSNN
cat > /tmp/train_model.py << 'EOF'
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import time

# Add project root to path (script is run from project root)
# __file__ is /tmp/train_model.py, so can't use it for path
sys.path.insert(0, os.getcwd())

# Import actual SNN modules
from src.model import SimpleSNN
from src.data import ECGDataset
from src.utils import set_seed, get_device

print("=" * 60)
print("NEUROMORPHIC SNN TRAINING")
print("=" * 60)
print()

# Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 50))
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
NUM_STEPS = 100  # Time steps for SNN
SEED = 42

print(f"📋 Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Time steps: {NUM_STEPS}")
print()

# Set seed for reproducibility
set_seed(SEED)

# Load data
print("📊 Loading data...")
data = torch.load('data/synthetic/mvp_dataset.pt')

# Create ECG datasets with spike encoding
train_dataset = ECGDataset(
    signals=data['train']['signals'].numpy(),
    labels=data['train']['labels'].numpy(),
    encode_spikes=True,
    num_steps=NUM_STEPS
)
val_dataset = ECGDataset(
    signals=data['val']['signals'].numpy(),
    labels=data['val']['labels'].numpy(),
    encode_spikes=True,
    num_steps=NUM_STEPS
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"   ✅ Train samples: {len(train_dataset)}")
print(f"   ✅ Val samples: {len(val_dataset)}")
print()

# Create SimpleSNN model
print("🧠 Creating SimpleSNN model...")

# Get input size from data
input_size = data['train']['signals'].shape[1]
model = SimpleSNN(
    input_size=input_size,
    hidden_size=128,
    output_size=2,
    beta=0.9
).to(DEVICE)

print(f"   ✅ Model created")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("🏋️  Training...")
print()

best_val_acc = 0.0
train_history = {'loss': [], 'acc': []}
val_history = {'loss': [], 'acc': []}

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()

        # SNN returns (spikes, membrane)
        spikes, membrane = model(batch_x)

        # Sum spikes over time dimension for classification
        outputs = spikes.sum(dim=0)  # [batch, classes]

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()

        pbar.set_postfix({
            'loss': f'{train_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * train_correct / train_total:.2f}%'
        })

    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # SNN returns (spikes, membrane)
            spikes, membrane = model(batch_x)

            # Sum spikes over time dimension for classification
            outputs = spikes.sum(dim=0)  # [batch, classes]

            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total

    # Save history
    train_history['loss'].append(train_loss)
    train_history['acc'].append(train_acc)
    val_history['loss'].append(val_loss)
    val_history['acc'].append(val_acc)

    print(f"   Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        Path('models').mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'models/best_model.pt')
        print(f"   💾 Saved best model (Val Acc: {val_acc:.2f}%)")

print()
print("=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print()
print(f"📊 Results:")
print(f"   Best validation accuracy: {best_val_acc:.2f}%")
print(f"   Model saved: models/best_model.pt")
print()

# Save training history
Path('results/metrics').mkdir(parents=True, exist_ok=True)
with open('results/metrics/training_history.json', 'w') as f:
    json.dump({
        'train': train_history,
        'val': val_history,
        'best_val_acc': best_val_acc
    }, f, indent=2)

print("📝 Training history saved: results/metrics/training_history.json")
print()

EOF

# Run training
$PYTHON /tmp/train_model.py

# Clean up
rm /tmp/train_model.py

echo ""
echo "🎉 ============================================"
echo "🎉 TRAINING COMPLETE!"
echo "🎉 ============================================"
echo ""
echo "📝 Generated Files:"
echo "   models/best_model.pt"
echo "   results/metrics/training_history.json"
echo ""
echo "📝 Next Steps:"
echo "   1. Evaluate model: python src/inference.py"
echo "   2. Launch demo: bash scripts/04_run_demo.sh"
echo "   3. View metrics: cat results/metrics/training_history.json"
echo ""
