#!/bin/bash

# ============================================
# CortexCore - DATA GENERATION
# ============================================
# Generates synthetic ECG/EEG data for MVP development
# Usage: bash scripts/02_generate_mvp_data.sh
# Time: ~1-2 minutes

set -e  # Exit on error

echo "📊 ============================================"
echo "📊 SYNTHETIC DATA GENERATION"
echo "📊 ============================================"
echo ""

# ==========================================
# Configuration
# ==========================================

# Load from .env if exists, otherwise use defaults
if [ -f ".env" ]; then
    source .env
fi

# Set defaults if not in .env
NUM_TRAIN_SAMPLES=${NUM_TRAIN_SAMPLES:-1000}
NUM_VAL_SAMPLES=${NUM_VAL_SAMPLES:-200}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-200}
SIGNAL_DURATION=${SIGNAL_DURATION:-10}
SAMPLING_RATE=${SAMPLING_RATE:-250}

echo "📋 Configuration:"
echo "   Training samples: $NUM_TRAIN_SAMPLES"
echo "   Validation samples: $NUM_VAL_SAMPLES"
echo "   Test samples: $NUM_TEST_SAMPLES"
echo "   Signal duration: ${SIGNAL_DURATION}s"
echo "   Sampling rate: ${SAMPLING_RATE}Hz"
echo ""

# ==========================================
# Check Dependencies
# ==========================================

echo "🔍 Checking dependencies..."

python -c "import neurokit2; import torch; import numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All required packages available"
else
    echo "   ❌ ERROR: Missing dependencies"
    echo "   Please run: bash scripts/01_setup_environment.sh"
    exit 1
fi

echo ""

# ==========================================
# Generate Data
# ==========================================

echo "🔬 Generating synthetic data..."
echo "   This may take 1-2 minutes..."
echo ""

# Create Python script for data generation
cat > /tmp/generate_data.py << 'EOF'
import sys
import os
import numpy as np
import torch
import neurokit2 as nk
from tqdm import tqdm
import json
from pathlib import Path

# Configuration from environment
NUM_TRAIN = int(os.getenv('NUM_TRAIN_SAMPLES', 1000))
NUM_VAL = int(os.getenv('NUM_VAL_SAMPLES', 200))
NUM_TEST = int(os.getenv('NUM_TEST_SAMPLES', 200))
DURATION = int(os.getenv('SIGNAL_DURATION', 10))
SAMPLING_RATE = int(os.getenv('SAMPLING_RATE', 250))

# Output paths
DATA_DIR = Path('data/synthetic')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_ecg_sample(condition='normal', duration=10, sampling_rate=250):
    """Generate synthetic ECG signal"""
    try:
        if condition == 'normal':
            # Normal ECG
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=70,
                noise=0.05
            )
        elif condition == 'arrhythmia':
            # Arrhythmia (irregular heart rate)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=120,
                noise=0.1
            )
            # Add irregularities
            irregularities = np.random.normal(0, 0.2, len(ecg))
            ecg = ecg + irregularities
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Normalize
        ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

        return ecg

    except Exception as e:
        print(f"Error generating ECG: {e}")
        # Return zeros as fallback
        return np.zeros(duration * sampling_rate)

def generate_dataset(n_samples, name='train'):
    """Generate dataset with balanced classes"""
    print(f"Generating {name} set ({n_samples} samples)...")

    signals = []
    labels = []

    # Balance classes
    n_per_class = n_samples // 2

    # Generate normal samples
    for i in tqdm(range(n_per_class), desc="Normal"):
        signal = generate_ecg_sample('normal', DURATION, SAMPLING_RATE)
        signals.append(signal)
        labels.append(0)  # Normal = 0

    # Generate arrhythmia samples
    for i in tqdm(range(n_per_class), desc="Arrhythmia"):
        signal = generate_ecg_sample('arrhythmia', DURATION, SAMPLING_RATE)
        signals.append(signal)
        labels.append(1)  # Arrhythmia = 1

    # Convert to tensors
    signals = torch.FloatTensor(np.array(signals))
    labels = torch.LongTensor(labels)

    # Shuffle
    indices = torch.randperm(len(signals))
    signals = signals[indices]
    labels = labels[indices]

    return signals, labels

try:
    print("=" * 50)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 50)
    print()

    # Generate datasets
    train_signals, train_labels = generate_dataset(NUM_TRAIN, 'train')
    val_signals, val_labels = generate_dataset(NUM_VAL, 'validation')
    test_signals, test_labels = generate_dataset(NUM_TEST, 'test')

    # Save datasets
    print("\nSaving datasets...")

    torch.save({
        'signals': train_signals,
        'labels': train_labels,
        'metadata': {
            'n_samples': len(train_signals),
            'duration': DURATION,
            'sampling_rate': SAMPLING_RATE,
            'n_classes': 2,
            'class_names': ['Normal', 'Arrhythmia']
        }
    }, DATA_DIR / 'train_data.pt')
    print(f"   ✅ Train: {len(train_signals)} samples")

    torch.save({
        'signals': val_signals,
        'labels': val_labels,
        'metadata': {
            'n_samples': len(val_signals),
            'duration': DURATION,
            'sampling_rate': SAMPLING_RATE,
            'n_classes': 2,
            'class_names': ['Normal', 'Arrhythmia']
        }
    }, DATA_DIR / 'val_data.pt')
    print(f"   ✅ Validation: {len(val_signals)} samples")

    torch.save({
        'signals': test_signals,
        'labels': test_labels,
        'metadata': {
            'n_samples': len(test_signals),
            'duration': DURATION,
            'sampling_rate': SAMPLING_RATE,
            'n_classes': 2,
            'class_names': ['Normal', 'Arrhythmia']
        }
    }, DATA_DIR / 'test_data.pt')
    print(f"   ✅ Test: {len(test_signals)} samples")

    # Save combined MVP dataset
    torch.save({
        'train': {'signals': train_signals, 'labels': train_labels},
        'val': {'signals': val_signals, 'labels': val_labels},
        'test': {'signals': test_signals, 'labels': test_labels},
        'metadata': {
            'duration': DURATION,
            'sampling_rate': SAMPLING_RATE,
            'n_classes': 2,
            'class_names': ['Normal', 'Arrhythmia'],
            'total_samples': len(train_signals) + len(val_signals) + len(test_signals)
        }
    }, DATA_DIR / 'mvp_dataset.pt')
    print(f"   ✅ MVP dataset: data/synthetic/mvp_dataset.pt")

    # Save statistics
    stats = {
        'train': {
            'n_samples': len(train_signals),
            'signal_shape': list(train_signals.shape),
            'class_distribution': {
                'normal': int((train_labels == 0).sum()),
                'arrhythmia': int((train_labels == 1).sum())
            }
        },
        'val': {
            'n_samples': len(val_signals),
            'class_distribution': {
                'normal': int((val_labels == 0).sum()),
                'arrhythmia': int((val_labels == 1).sum())
            }
        },
        'test': {
            'n_samples': len(test_signals),
            'class_distribution': {
                'normal': int((test_labels == 0).sum()),
                'arrhythmia': int((test_labels == 1).sum())
            }
        }
    }

    with open(DATA_DIR / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Statistics: data/synthetic/dataset_stats.json")

    print()
    print("=" * 50)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 50)
    print()
    print("📊 Dataset Summary:")
    print(f"   Total samples: {stats['train']['n_samples'] + stats['val']['n_samples'] + stats['test']['n_samples']}")
    print(f"   Signal shape: {stats['train']['signal_shape']}")
    print(f"   Classes: Normal ({stats['train']['class_distribution']['normal']} train), "
          f"Arrhythmia ({stats['train']['class_distribution']['arrhythmia']} train)")
    print()

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run data generation
python /tmp/generate_data.py

# Clean up
rm /tmp/generate_data.py

echo ""
echo "🎉 ============================================"
echo "🎉 DATA GENERATION COMPLETE!"
echo "🎉 ============================================"
echo ""
echo "📝 Generated Files:"
echo "   data/synthetic/train_data.pt"
echo "   data/synthetic/val_data.pt"
echo "   data/synthetic/test_data.pt"
echo "   data/synthetic/mvp_dataset.pt (combined)"
echo "   data/synthetic/dataset_stats.json"
echo ""
echo "📝 Next Steps:"
echo "   1. Inspect data: jupyter notebook notebooks/02_data_generation.ipynb"
echo "   2. Train model: bash scripts/03_train_mvp_model.sh"
echo ""
