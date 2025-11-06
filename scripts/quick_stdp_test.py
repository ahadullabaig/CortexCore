"""
Quick STDP Training Test
========================

This script runs a mini 3-phase training (2+3+2 epochs) to validate
the complete STDP pipeline works end-to-end before running the full
70-epoch training.

Phases:
- Phase 1: 2 epochs STDP (instead of 20)
- Phase 2: 3 epochs Hybrid (instead of 30)
- Phase 3: 2 epochs Finetune (instead of 20)
Total: 7 epochs (~5-10 minutes on GPU)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.model import HybridSTDP_SNN
from src.stdp import STDPConfig
from src.train_stdp import train_three_phase
from src.data import load_dataset
from src.utils import set_seed

def main():
    print("=" * 80)
    print("🧪 QUICK STDP PIPELINE TEST")
    print("=" * 80)
    print("Running mini 3-phase training: 2+3+2 epochs = 7 total")
    print("This validates the pipeline before full 70-epoch training")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(42)

    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    DATA_DIR = 'data/synthetic'

    print(f"\n⚙️  Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Data Directory: {DATA_DIR}")

    # Load data
    print(f"\n📊 Loading data...")
    try:
        train_loader = load_dataset(
            f'{DATA_DIR}/train_data.pt',
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = load_dataset(
            f'{DATA_DIR}/val_data.pt',
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Make sure you've run: bash scripts/02_generate_mvp_data.sh")
        return 1

    # Create STDP configuration
    print(f"\n🧠 Creating STDP configuration...")
    config = STDPConfig(
        # Core STDP
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.01,
        a_minus=0.01,
        # Homeostatic plasticity
        use_homeostasis=True,
        target_rate=10.0,
        homeostatic_scale=0.001,
        # Multi-timescale
        use_multiscale=True,
        tau_fast=10.0,
        tau_slow=100.0,
        alpha_initial=0.8,
        alpha_final=0.3
    )

    print(f"   STDP Enabled: ✓")
    print(f"   Homeostatic Plasticity: {config.use_homeostasis}")
    print(f"   Multi-Timescale: {config.use_multiscale}")
    print(f"   Fast/Slow: {config.tau_fast}ms / {config.tau_slow}ms")

    # Create model
    print(f"\n🏗️  Creating HybridSTDP_SNN model...")
    model = HybridSTDP_SNN(
        input_size=2500,
        hidden_size=128,
        output_size=2,
        config=config
    )

    from src.model import count_parameters
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Architecture: 2500 → 128 → 2")

    # Run quick 3-phase training
    print(f"\n🚀 Starting mini 3-phase training...")
    print("=" * 80)

    history = train_three_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        save_dir='models/stdp_test',
        # Mini phase durations
        stdp_epochs=2,       # Phase 1: Just 2 epochs
        hybrid_epochs=3,     # Phase 2: Just 3 epochs
        finetune_epochs=2,   # Phase 3: Just 2 epochs
        # Learning rates
        stdp_lr_scale=1.0,
        hybrid_lr=0.001,
        finetune_lr=0.0001,
        log_interval=1       # Log every epoch
    )

    print("=" * 80)
    print("✅ QUICK TEST COMPLETE!")
    print("=" * 80)

    # Summary
    print("\n📊 Training Summary:")
    print(f"\n   Phase 1 (STDP):")
    print(f"      Epochs: 2")
    final_stdp = history['phase1_stdp'][-1]
    print(f"      Final LTP/LTD Ratio: {final_stdp['ltp_ltd_ratio']:.3f}")
    print(f"      Final Alpha: {final_stdp['avg_alpha']:.3f}")

    print(f"\n   Phase 2 (Hybrid):")
    print(f"      Epochs: 3")
    final_hybrid = history['phase2_hybrid'][-1]
    print(f"      Final Train Accuracy: {final_hybrid['accuracy']:.2f}%")
    final_hybrid_val = history['val_phase2'][-1]
    print(f"      Final Val Accuracy: {final_hybrid_val['val_accuracy']:.2f}%")

    print(f"\n   Phase 3 (Finetune):")
    print(f"      Epochs: 2")
    final_finetune = history['phase3_finetune'][-1]
    print(f"      Final Train Accuracy: {final_finetune['accuracy']:.2f}%")
    final_finetune_val = history['val_phase3'][-1]
    print(f"      Final Val Accuracy: {final_finetune_val['val_accuracy']:.2f}%")

    print(f"\n   Total Epochs: 7")
    print(f"   Best Val Accuracy: {max(v['val_accuracy'] for v in history['val_phase2'] + history['val_phase3']):.2f}%")

    print("\n" + "=" * 80)
    print("🎉 PIPELINE VALIDATED - Ready for full 70-epoch training!")
    print("=" * 80)
    print("\nNext step: Run full training with:")
    print("   python scripts/train_full_stdp.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
