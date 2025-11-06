"""
STDP Visualization Module
==========================

Owner: CS4 / Deployment Engineer (with CS2 support)
Phase: Week 2 (Days 8-14)

This module provides comprehensive visualizations for STDP learning dynamics:
1. Weight evolution heatmaps (animated)
2. Spike raster plots with STDP windows
3. LTP/LTD event tracking
4. Feature map comparisons (t-SNE)
5. Homeostatic dashboard
6. Multi-timescale analysis

All visualizations are demo-ready and publication-quality.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# ============================================
# 1. Weight Evolution Visualization
# ============================================

def plot_weight_evolution(
    weight_history: List[torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "STDP Weight Evolution",
    vmin: float = 0.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Plot weight matrix evolution over time as heatmaps

    Args:
        weight_history: List of weight tensors [output, input] over epochs
        save_path: Path to save figure
        title: Plot title
        vmin/vmax: Color scale bounds

    Returns:
        fig: Matplotlib figure
    """
    n_snapshots = min(len(weight_history), 6)  # Show max 6 snapshots
    indices = np.linspace(0, len(weight_history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, epoch_idx in enumerate(indices):
        weights = weight_history[epoch_idx].cpu().numpy()

        im = axes[idx].imshow(
            weights,
            aspect='auto',
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        axes[idx].set_title(f'Epoch {epoch_idx + 1}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Input Neurons', fontsize=12)
        axes[idx].set_ylabel('Hidden Neurons', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('Weight', fontsize=11)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved weight evolution to {save_path}")

    return fig


def create_weight_evolution_animation(
    weight_history: List[torch.Tensor],
    save_path: str,
    fps: int = 5,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    """
    Create animated GIF of weight evolution

    Args:
        weight_history: List of weight tensors over epochs
        save_path: Path to save GIF
        fps: Frames per second
        vmin/vmax: Color scale bounds
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        weights = weight_history[frame].cpu().numpy()

        im = ax.imshow(
            weights,
            aspect='auto',
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        ax.set_title(f'STDP Weight Evolution - Epoch {frame + 1}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Input Neurons', fontsize=14)
        ax.set_ylabel('Hidden Neurons', fontsize=14)

        return [im]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(weight_history),
        interval=1000 // fps,
        blit=True
    )

    anim.save(save_path, writer='pillow', fps=fps, dpi=150)
    print(f"✅ Saved weight evolution animation to {save_path}")
    plt.close()


# ============================================
# 2. Spike Raster Plot with STDP Windows
# ============================================

def plot_spike_raster(
    spike_train: torch.Tensor,
    stdp_window: float = 20.0,
    highlight_events: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    title: str = "Spike Raster Plot with STDP Windows"
) -> plt.Figure:
    """
    Plot spike raster with STDP temporal windows

    Args:
        spike_train: Spikes [time_steps, batch, neurons]
        stdp_window: STDP time window (ms)
        highlight_events: List of (time, neuron) pairs to highlight
        save_path: Path to save figure
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    # Take first batch for visualization
    spikes = spike_train[:, 0, :].cpu().numpy()  # [time, neurons]
    time_steps, n_neurons = spikes.shape

    # Find spike times and neuron indices
    spike_times, spike_neurons = np.where(spikes > 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot spikes
    ax.scatter(
        spike_times,
        spike_neurons,
        s=5,
        c='black',
        marker='|',
        alpha=0.7,
        label='Spikes'
    )

    # Highlight STDP events if provided
    if highlight_events:
        for t, n in highlight_events:
            # Draw STDP window
            rect = Rectangle(
                (max(0, t - stdp_window), n - 0.5),
                2 * stdp_window,
                1,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                alpha=0.5
            )
            ax.add_patch(rect)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(0, time_steps)
    ax.set_ylim(-1, n_neurons)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved spike raster to {save_path}")

    return fig


# ============================================
# 3. LTP/LTD Event Tracking
# ============================================

def plot_ltp_ltd_evolution(
    history: Dict[str, List[Dict]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot LTP/LTD event counts over training

    Args:
        history: Training history dict from train_three_phase()
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    # Extract STDP phase data
    stdp_history = history.get('phase1_stdp', [])

    if not stdp_history:
        print("⚠️  No STDP history found")
        return None

    epochs = [h['epoch'] for h in stdp_history]
    ltp_fast = [h['avg_ltp_fast'] for h in stdp_history]
    ltd_fast = [h['avg_ltd_fast'] for h in stdp_history]
    ltp_slow = [h.get('total_ltp_slow', 0) / len(stdp_history) for h in stdp_history]
    ltd_slow = [h.get('total_ltd_slow', 0) / len(stdp_history) for h in stdp_history]
    ratio = [h['ltp_ltd_ratio'] for h in stdp_history]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: LTP/LTD counts
    axes[0].plot(epochs, ltp_fast, 'o-', color='green', linewidth=2, markersize=6, label='LTP (Fast)', alpha=0.8)
    axes[0].plot(epochs, ltd_fast, 's-', color='red', linewidth=2, markersize=6, label='LTD (Fast)', alpha=0.8)

    if any(ltp_slow) and any(ltd_slow):
        axes[0].plot(epochs, ltp_slow, 'o--', color='lightgreen', linewidth=2, markersize=4, label='LTP (Slow)', alpha=0.6)
        axes[0].plot(epochs, ltd_slow, 's--', color='lightcoral', linewidth=2, markersize=4, label='LTD (Slow)', alpha=0.6)

    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Event Count', fontsize=14)
    axes[0].set_title('LTP/LTD Event Evolution', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: LTP/LTD ratio
    axes[1].plot(epochs, ratio, 'o-', color='purple', linewidth=2, markersize=6, alpha=0.8)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Balanced (1:1)')
    axes[1].fill_between(epochs, 0.9, 1.1, color='gray', alpha=0.2, label='Healthy range')

    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('LTP/LTD Ratio', fontsize=14)
    axes[1].set_title('LTP/LTD Balance', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved LTP/LTD evolution to {save_path}")

    return fig


# ============================================
# 4. Multi-Timescale Analysis
# ============================================

def plot_multiscale_analysis(
    history: Dict[str, List[Dict]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multi-timescale STDP analysis (fast vs slow weights)

    Args:
        history: Training history dict
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    stdp_history = history.get('phase1_stdp', [])

    if not stdp_history:
        print("⚠️  No STDP history found")
        return None

    epochs = [h['epoch'] for h in stdp_history]
    alpha = [h['avg_alpha'] for h in stdp_history]
    divergence = [h['avg_divergence'] for h in stdp_history]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Alpha annealing
    axes[0].plot(epochs, alpha, 'o-', color='blue', linewidth=2, markersize=6, alpha=0.8)
    axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, label='Initial (favor fast)', alpha=0.6)
    axes[0].axhline(y=0.3, color='orange', linestyle='--', linewidth=1.5, label='Final (favor slow)', alpha=0.6)

    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Alpha (Fast Weight)', fontsize=14)
    axes[0].set_title('Multi-Timescale Alpha Annealing', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Plot 2: Weight divergence
    axes[1].plot(epochs, divergence, 'o-', color='red', linewidth=2, markersize=6, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Fast-Slow Weight Divergence', fontsize=14)
    axes[1].set_title('Fast vs Slow Weight Divergence', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved multi-timescale analysis to {save_path}")

    return fig


# ============================================
# 5. Homeostatic Dashboard
# ============================================

def plot_homeostatic_dashboard(
    firing_rates: List[torch.Tensor],
    weight_distributions: List[torch.Tensor],
    target_rate: float = 10.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot homeostatic plasticity metrics

    Args:
        firing_rates: List of firing rate tensors [neurons] over epochs
        weight_distributions: List of weight tensors over epochs
        target_rate: Target firing rate (Hz)
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Firing rate distribution (first/middle/last epochs)
    epochs_to_plot = [0, len(firing_rates) // 2, len(firing_rates) - 1]
    colors = ['blue', 'green', 'red']
    labels = ['Epoch 1', f'Epoch {len(firing_rates) // 2}', f'Epoch {len(firing_rates)}']

    for epoch_idx, color, label in zip(epochs_to_plot, colors, labels):
        rates = firing_rates[epoch_idx].cpu().numpy()
        axes[0, 0].hist(rates, bins=30, alpha=0.5, color=color, label=label, edgecolor='black')

    axes[0, 0].axvline(x=target_rate, color='black', linestyle='--', linewidth=2, label=f'Target: {target_rate} Hz')
    axes[0, 0].set_xlabel('Firing Rate (Hz)', fontsize=12)
    axes[0, 0].set_ylabel('Neuron Count', fontsize=12)
    axes[0, 0].set_title('Firing Rate Distributions', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean firing rate evolution
    mean_rates = [fr.mean().item() for fr in firing_rates]
    std_rates = [fr.std().item() for fr in firing_rates]
    epochs = list(range(len(firing_rates)))

    axes[0, 1].plot(epochs, mean_rates, 'o-', color='blue', linewidth=2, markersize=5)
    axes[0, 1].fill_between(
        epochs,
        np.array(mean_rates) - np.array(std_rates),
        np.array(mean_rates) + np.array(std_rates),
        alpha=0.3,
        color='blue'
    )
    axes[0, 1].axhline(y=target_rate, color='black', linestyle='--', linewidth=2, label=f'Target: {target_rate} Hz')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
    axes[0, 1].set_title('Firing Rate Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Weight distribution evolution
    for epoch_idx, color, label in zip(epochs_to_plot, colors, labels):
        weights = weight_distributions[epoch_idx].cpu().numpy().flatten()
        axes[1, 0].hist(weights, bins=50, alpha=0.5, color=color, label=label, edgecolor='black')

    axes[1, 0].set_xlabel('Weight Value', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Weight Distributions', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Saturation metric
    saturation_ratios = []
    for weights in weight_distributions:
        w = weights.cpu().numpy().flatten()
        saturated = ((w <= 0.05).sum() + (w >= 0.95).sum()) / len(w)
        saturation_ratios.append(saturated * 100)

    axes[1, 1].plot(epochs, saturation_ratios, 'o-', color='red', linewidth=2, markersize=5)
    axes[1, 1].axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Warning threshold (10%)')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Saturation Ratio (%)', fontsize=12)
    axes[1, 1].set_title('Weight Saturation Monitor', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 100)

    plt.suptitle('Homeostatic Plasticity Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved homeostatic dashboard to {save_path}")

    return fig


# ============================================
# 6. Complete Training Summary Dashboard
# ============================================

def plot_training_summary(
    history: Dict[str, List[Dict]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive training summary across all 3 phases

    Args:
        history: Complete training history from train_three_phase()
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract data
    stdp_history = history.get('phase1_stdp', [])
    hybrid_history = history.get('phase2_hybrid', [])
    finetune_history = history.get('phase3_finetune', [])
    val_phase1 = history.get('val_phase1', [])
    val_phase2 = history.get('val_phase2', [])
    val_phase3 = history.get('val_phase3', [])

    # Plot 1: STDP LTP/LTD ratio (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if stdp_history:
        epochs = [h['epoch'] for h in stdp_history]
        ratio = [h['ltp_ltd_ratio'] for h in stdp_history]
        ax1.plot(epochs, ratio, 'o-', color='purple', linewidth=2, markersize=5)
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
        ax1.set_title('Phase 1: LTP/LTD Ratio', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Ratio')
        ax1.grid(True, alpha=0.3)

    # Plot 2: Alpha annealing (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if stdp_history:
        epochs = [h['epoch'] for h in stdp_history]
        alpha = [h['avg_alpha'] for h in stdp_history]
        ax2.plot(epochs, alpha, 'o-', color='blue', linewidth=2, markersize=5)
        ax2.set_title('Phase 1: Multi-Timescale Alpha', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Alpha')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy evolution (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    all_epochs = []
    all_val_acc = []

    # Phase 1 validation
    if val_phase1:
        all_epochs.extend([v['epoch'] for v in val_phase1])
        all_val_acc.extend([v['val_accuracy'] for v in val_phase1])

    # Phase 2 validation
    if val_phase2:
        all_epochs.extend([v['epoch'] for v in val_phase2])
        all_val_acc.extend([v['val_accuracy'] for v in val_phase2])

    # Phase 3 validation
    if val_phase3:
        all_epochs.extend([v['epoch'] for v in val_phase3])
        all_val_acc.extend([v['val_accuracy'] for v in val_phase3])

    if all_epochs:
        ax3.plot(all_epochs, all_val_acc, 'o-', color='green', linewidth=2, markersize=5)
        ax3.axhline(y=85, color='orange', linestyle='--', linewidth=1.5, label='Target: 85%')
        ax3.axhline(y=92, color='red', linestyle='--', linewidth=1.5, label='Goal: 92%')
        ax3.set_title('Validation Accuracy Evolution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Global Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Phase 2 training loss (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    if hybrid_history:
        epochs = [h['epoch'] for h in hybrid_history]
        loss = [h['loss'] for h in hybrid_history]
        ax4.plot(epochs, loss, 'o-', color='orange', linewidth=2, markersize=5)
        ax4.set_title('Phase 2: Training Loss', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Global Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)

    # Plot 5: Phase 2 training accuracy (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    if hybrid_history:
        epochs = [h['epoch'] for h in hybrid_history]
        acc = [h['accuracy'] for h in hybrid_history]
        ax5.plot(epochs, acc, 'o-', color='blue', linewidth=2, markersize=5)
        ax5.set_title('Phase 2: Training Accuracy', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Global Epoch')
        ax5.set_ylabel('Accuracy (%)')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Phase 3 training accuracy (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    if finetune_history:
        epochs = [h['epoch'] for h in finetune_history]
        acc = [h['accuracy'] for h in finetune_history]
        ax6.plot(epochs, acc, 'o-', color='red', linewidth=2, markersize=5)
        ax6.set_title('Phase 3: Fine-tuning Accuracy', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Global Epoch')
        ax6.set_ylabel('Accuracy (%)')
        ax6.grid(True, alpha=0.3)

    # Plot 7: Phase comparison bar chart (bottom span)
    ax7 = fig.add_subplot(gs[2, :])
    phases = ['Phase 1\n(STDP)', 'Phase 2\n(Hybrid)', 'Phase 3\n(Finetune)']
    val_accs = []

    if val_phase1 and len(val_phase1) > 0:
        val_accs.append(val_phase1[-1]['val_accuracy'])
    else:
        val_accs.append(0)

    if val_phase2 and len(val_phase2) > 0:
        val_accs.append(val_phase2[-1]['val_accuracy'])
    else:
        val_accs.append(0)

    if val_phase3 and len(val_phase3) > 0:
        val_accs.append(val_phase3[-1]['val_accuracy'])
    else:
        val_accs.append(0)

    colors = ['purple', 'orange', 'red']
    bars = ax7.bar(phases, val_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, val_accs):
        height = bar.get_height()
        ax7.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{val:.2f}%',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    ax7.axhline(y=85, color='green', linestyle='--', linewidth=2, label='MVP Target: 85%', alpha=0.7)
    ax7.axhline(y=92, color='blue', linestyle='--', linewidth=2, label='Phase 2 Goal: 92%', alpha=0.7)
    ax7.set_title('Final Validation Accuracy by Phase', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax7.set_ylim(0, 100)
    ax7.legend(fontsize=11, loc='upper left')
    ax7.grid(True, alpha=0.3, axis='y')

    plt.suptitle('STDP Training Summary Dashboard', fontsize=20, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training summary to {save_path}")

    return fig


# ============================================
# 7. Quick Visualization Generator
# ============================================

def generate_all_visualizations(
    history_path: str,
    weight_history_path: Optional[str] = None,
    output_dir: str = 'results/stdp_visualizations'
) -> None:
    """
    Generate all STDP visualizations from training history

    Args:
        history_path: Path to training history JSON
        weight_history_path: Path to saved weight history (optional)
        output_dir: Output directory for visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 GENERATING STDP VISUALIZATIONS")
    print("=" * 80)

    # Load history
    print(f"\n📖 Loading training history from {history_path}...")
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")

    # 1. Training summary dashboard
    print("   1. Training summary dashboard...")
    plot_training_summary(
        history,
        save_path=str(output_path / 'training_summary.png')
    )
    plt.close('all')

    # 2. LTP/LTD evolution
    print("   2. LTP/LTD evolution...")
    plot_ltp_ltd_evolution(
        history,
        save_path=str(output_path / 'ltp_ltd_evolution.png')
    )
    plt.close('all')

    # 3. Multi-timescale analysis
    print("   3. Multi-timescale analysis...")
    plot_multiscale_analysis(
        history,
        save_path=str(output_path / 'multiscale_analysis.png')
    )
    plt.close('all')

    # 4. Weight evolution (if available)
    if weight_history_path and Path(weight_history_path).exists():
        print("   4. Weight evolution...")
        weight_history = torch.load(weight_history_path)
        plot_weight_evolution(
            weight_history,
            save_path=str(output_path / 'weight_evolution.png')
        )
        plt.close('all')

    print("\n✅ All visualizations generated!")
    print(f"   Output directory: {output_dir}")
    print("=" * 80)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 STDP Visualization Module Loaded")
    print("\nAvailable functions:")
    print("   - plot_weight_evolution()")
    print("   - plot_spike_raster()")
    print("   - plot_ltp_ltd_evolution()")
    print("   - plot_multiscale_analysis()")
    print("   - plot_homeostatic_dashboard()")
    print("   - plot_training_summary()")
    print("   - generate_all_visualizations()")
    print("\nExample:")
    print("   generate_all_visualizations('models/stdp/stdp_training_history.json')")
