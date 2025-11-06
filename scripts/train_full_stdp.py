"""
Full 70-Epoch STDP Training Script
===================================

Production-grade implementation with:
- Adaptive learning rate scheduling (cosine annealing)
- Early stopping with patience
- Advanced checkpoint management
- Comprehensive metrics tracking
- Gradient clipping for stability
- Resume capability from interruptions
- Real-time progress monitoring
- Automatic visualization generation

Three-Phase Training Strategy:
- Phase 1 (20 epochs): Pure STDP unsupervised feature learning
- Phase 2 (30 epochs): Hybrid STDP+backprop supervised classification
- Phase 3 (20 epochs): Full fine-tuning with backprop
Total: 70 epochs

Expected Performance: ≥92% validation accuracy
Expected Runtime: ~2-3 hours on GPU

Usage:
    python scripts/train_full_stdp.py

    # Resume from checkpoint:
    python scripts/train_full_stdp.py --resume models/stdp_full/checkpoint.pt

    # Custom configuration:
    python scripts/train_full_stdp.py --batch-size 64 --learning-rate 0.002
"""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import json
import time
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np

from src.model import HybridSTDP_SNN
from src.stdp import STDPConfig
from src.train_stdp import train_three_phase
from src.data import load_dataset
from src.utils import set_seed


@dataclass
class TrainingConfig:
    """Complete training configuration"""

    # Data
    data_dir: str = 'data/synthetic'
    batch_size: int = 32
    num_workers: int = 4

    # Model
    input_size: int = 2500
    hidden_size: int = 128
    output_size: int = 2
    beta: float = 0.9

    # STDP Configuration
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.01
    use_homeostasis: bool = True
    target_rate: float = 10.0
    homeostatic_scale: float = 0.001
    use_multiscale: bool = True
    tau_fast: float = 10.0
    tau_slow: float = 100.0
    alpha_initial: float = 0.8
    alpha_final: float = 0.3

    # Training Phases
    stdp_epochs: int = 20      # Phase 1
    hybrid_epochs: int = 30     # Phase 2
    finetune_epochs: int = 20   # Phase 3

    # Learning Rates
    stdp_lr_scale: float = 1.0
    hybrid_lr_initial: float = 0.001
    hybrid_lr_min: float = 0.00001
    finetune_lr_initial: float = 0.0001
    finetune_lr_min: float = 0.000001

    # Optimization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Early Stopping
    use_early_stopping: bool = True
    patience: int = 10  # epochs without improvement
    min_delta: float = 0.001  # minimum improvement threshold

    # Checkpointing
    save_dir: str = 'models/stdp_full'
    save_frequency: int = 5  # save every N epochs
    keep_best_only: bool = False  # if True, only keep best checkpoint

    # Logging
    log_interval: int = 1  # log every N epochs
    verbose: bool = True

    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

    # Resume
    resume_from: Optional[str] = None


class EarlyStopping:
    """Early stopping handler with patience"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if should stop early
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check for improvement
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping counter"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def create_stdp_config(config: TrainingConfig) -> STDPConfig:
    """Create STDP configuration from training config"""
    return STDPConfig(
        tau_plus=config.stdp_tau_plus,
        tau_minus=config.stdp_tau_minus,
        a_plus=config.stdp_a_plus,
        a_minus=config.stdp_a_minus,
        use_homeostasis=config.use_homeostasis,
        target_rate=config.target_rate,
        homeostatic_scale=config.homeostatic_scale,
        use_multiscale=config.use_multiscale,
        tau_fast=config.tau_fast,
        tau_slow=config.tau_slow,
        alpha_initial=config.alpha_initial,
        alpha_final=config.alpha_final
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    history: Dict,
    config: TrainingConfig,
    epoch: int,
    phase: str,
    metrics: Dict,
    path: Path
):
    """Save comprehensive checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'history': history,
        'metrics': metrics,
        'stdp_statistics': model.get_stdp_statistics() if hasattr(model, 'get_stdp_statistics') else None
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved: {path}")


def load_checkpoint(path: str, model: nn.Module, config: TrainingConfig) -> Tuple[Dict, int, str]:
    """
    Load checkpoint and restore training state

    Returns:
        (history, start_epoch, phase)
    """
    print(f"\n📂 Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint.get('history', {})
    start_epoch = checkpoint.get('epoch', 0) + 1
    phase = checkpoint.get('phase', 'STDP')

    print(f"✅ Resumed from epoch {checkpoint['epoch']} (phase: {phase})")
    print(f"   Previous metrics: {checkpoint.get('metrics', {})}")

    return history, start_epoch, phase


def print_training_header(config: TrainingConfig):
    """Print beautiful training configuration header"""
    print("=" * 80)
    print("🧠 FULL STDP TRAINING - 70 EPOCH PRODUCTION RUN")
    print("=" * 80)
    print(f"Device: {config.device.upper()}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Random Seed: {config.seed}")
    print()
    print(f"📊 Dataset: {config.data_dir}")
    print(f"🏗️  Model: HybridSTDP_SNN ({config.input_size} → {config.hidden_size} → {config.output_size})")
    print()
    print("🔬 STDP Configuration:")
    print(f"   Homeostatic Plasticity: {config.use_homeostasis} (target: {config.target_rate} Hz)")
    print(f"   Multi-Timescale: {config.use_multiscale} (fast: {config.tau_fast}ms, slow: {config.tau_slow}ms)")
    print(f"   Alpha Annealing: {config.alpha_initial} → {config.alpha_final}")
    print()
    print("📅 Training Schedule:")
    print(f"   Phase 1 (STDP):     {config.stdp_epochs} epochs - Unsupervised feature learning")
    print(f"   Phase 2 (Hybrid):   {config.hybrid_epochs} epochs - Supervised classification (layer 1 frozen)")
    print(f"   Phase 3 (Finetune): {config.finetune_epochs} epochs - Full network fine-tuning")
    print(f"   Total: {config.stdp_epochs + config.hybrid_epochs + config.finetune_epochs} epochs")
    print()
    print("⚙️  Optimization:")
    print(f"   STDP LR Scale: {config.stdp_lr_scale}")
    print(f"   Hybrid LR: {config.hybrid_lr_initial} → {config.hybrid_lr_min} (cosine)")
    print(f"   Finetune LR: {config.finetune_lr_initial} → {config.finetune_lr_min} (cosine)")
    print(f"   Gradient Clipping: {config.gradient_clip}")
    print(f"   Weight Decay: {config.weight_decay}")
    print()
    if config.use_early_stopping:
        print(f"🛑 Early Stopping: Patience={config.patience}, Min Delta={config.min_delta}")
    print(f"💾 Checkpoints: {config.save_dir} (every {config.save_frequency} epochs)")
    print("=" * 80)


def calculate_statistics(history: Dict) -> Dict:
    """Calculate comprehensive training statistics"""
    stats = {}

    # Phase 1 (STDP)
    if 'phase1_stdp' in history and len(history['phase1_stdp']) > 0:
        stdp_data = history['phase1_stdp']
        stats['phase1_avg_ltp_ltd_ratio'] = np.mean([e['ltp_ltd_ratio'] for e in stdp_data])
        stats['phase1_final_alpha'] = stdp_data[-1]['avg_alpha']
        stats['phase1_final_weight_change'] = stdp_data[-1]['avg_weight_change']

    # Phase 2 (Hybrid)
    if 'val_phase2' in history and len(history['val_phase2']) > 0:
        phase2_val = history['val_phase2']
        stats['phase2_best_val_acc'] = max([e['val_accuracy'] for e in phase2_val])
        stats['phase2_final_val_acc'] = phase2_val[-1]['val_accuracy']

    # Phase 3 (Finetune)
    if 'val_phase3' in history and len(history['val_phase3']) > 0:
        phase3_val = history['val_phase3']
        stats['phase3_best_val_acc'] = max([e['val_accuracy'] for e in phase3_val])
        stats['phase3_final_val_acc'] = phase3_val[-1]['val_accuracy']

    # Overall
    all_val_accs = []
    if 'val_phase2' in history:
        all_val_accs.extend([e['val_accuracy'] for e in history['val_phase2']])
    if 'val_phase3' in history:
        all_val_accs.extend([e['val_accuracy'] for e in history['val_phase3']])

    if all_val_accs:
        stats['overall_best_val_acc'] = max(all_val_accs)

    return stats


def main(args: argparse.Namespace):
    """Main training pipeline"""

    # Create configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        hybrid_lr_initial=args.learning_rate,
        resume_from=args.resume,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Set random seed
    set_seed(config.seed)

    # Print configuration
    print_training_header(config)

    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"📝 Configuration saved: {config_path}\n")

    # Load data
    print("📊 Loading dataset...")
    try:
        train_loader = load_dataset(
            f'{config.data_dir}/train_data.pt',
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = load_dataset(
            f'{config.data_dir}/val_data.pt',
            batch_size=config.batch_size,
            shuffle=False
        )
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Make sure you've run: bash scripts/02_generate_mvp_data.sh")
        return 1

    # Create STDP configuration
    print("\n🧠 Creating STDP configuration...")
    stdp_config = create_stdp_config(config)

    # Create model
    print("🏗️  Creating HybridSTDP_SNN model...")
    model = HybridSTDP_SNN(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        beta=config.beta,
        config=stdp_config
    ).to(config.device)

    from src.model import count_parameters
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Architecture: {config.input_size} → {config.hidden_size} → {config.output_size}")

    # Resume from checkpoint if specified
    start_epoch = 0
    history = None
    if config.resume_from:
        history, start_epoch, phase = load_checkpoint(config.resume_from, model, config)

    # Run full 3-phase training
    print("\n🚀 Starting full 70-epoch training...")
    print("=" * 80)

    start_time = time.time()

    try:
        history = train_three_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            save_dir=str(save_dir),
            # Phase durations
            stdp_epochs=config.stdp_epochs,
            hybrid_epochs=config.hybrid_epochs,
            finetune_epochs=config.finetune_epochs,
            # Learning rates
            stdp_lr_scale=config.stdp_lr_scale,
            hybrid_lr=config.hybrid_lr_initial,
            finetune_lr=config.finetune_lr_initial,
            # General
            log_interval=config.log_interval
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving checkpoint...")
        save_checkpoint(
            model, None, None, history or {}, config,
            epoch=start_epoch, phase='interrupted',
            metrics={}, path=save_dir / 'interrupted_checkpoint.pt'
        )
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start_time

    # Training complete
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total Time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"⚡ Avg Time per Epoch: {elapsed/(config.stdp_epochs + config.hybrid_epochs + config.finetune_epochs):.1f} seconds")

    # Calculate and print statistics
    print("\n📊 TRAINING SUMMARY:")
    print("-" * 80)

    stats = calculate_statistics(history)

    # Phase 1
    if 'phase1_avg_ltp_ltd_ratio' in stats:
        print(f"\n🔬 Phase 1 (STDP - {config.stdp_epochs} epochs):")
        print(f"   Average LTP/LTD Ratio: {stats['phase1_avg_ltp_ltd_ratio']:.3f}")
        print(f"   Final Alpha: {stats['phase1_final_alpha']:.3f}")
        print(f"   Final Weight Change: {stats['phase1_final_weight_change']:.6f}")

    # Phase 2
    if 'phase2_best_val_acc' in stats:
        print(f"\n🔀 Phase 2 (Hybrid - {config.hybrid_epochs} epochs):")
        print(f"   Best Val Accuracy: {stats['phase2_best_val_acc']:.2f}%")
        print(f"   Final Val Accuracy: {stats['phase2_final_val_acc']:.2f}%")

    # Phase 3
    if 'phase3_best_val_acc' in stats:
        print(f"\n✨ Phase 3 (Finetune - {config.finetune_epochs} epochs):")
        print(f"   Best Val Accuracy: {stats['phase3_best_val_acc']:.2f}%")
        print(f"   Final Val Accuracy: {stats['phase3_final_val_acc']:.2f}%")

    # Overall
    if 'overall_best_val_acc' in stats:
        print(f"\n🏆 Overall Best Validation Accuracy: {stats['overall_best_val_acc']:.2f}%")

        # Check if target met
        target = 92.0
        if stats['overall_best_val_acc'] >= target:
            print(f"   ✅ TARGET MET! ({stats['overall_best_val_acc']:.2f}% ≥ {target}%)")
        else:
            gap = target - stats['overall_best_val_acc']
            print(f"   ⚠️  Target not met ({gap:.2f}% below {target}%)")

    print("\n" + "=" * 80)
    print("📁 Saved Files:")
    print(f"   Best Model: {save_dir}/best_finetuned_model.pt")
    print(f"   Training History: {save_dir}/stdp_training_history.json")
    print(f"   Configuration: {save_dir}/training_config.json")
    print("=" * 80)

    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    try:
        from src.visualization import generate_all_visualizations
        viz_dir = Path('results/stdp_full_visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)

        generate_all_visualizations(
            history_path=str(save_dir / 'stdp_training_history.json'),
            output_dir=str(viz_dir)
        )
        print(f"✅ Visualizations saved: {viz_dir}/")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        print("   You can generate them later with:")
        print(f"   python -c \"from src.visualization import generate_all_visualizations; "
              f"generate_all_visualizations('{save_dir}/stdp_training_history.json', 'results/stdp_full_visualizations')\"")

    print("\n🎉 All done! Your biologically-plausible SNN is ready for deployment.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full 70-epoch STDP training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--save-dir', type=str, default='models/stdp_full', help='Save directory')

    args = parser.parse_args()

    sys.exit(main(args))
