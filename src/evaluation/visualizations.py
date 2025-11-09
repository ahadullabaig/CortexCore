"""
Visualization Module
====================

Plot generation for error analysis, performance metrics, and robustness testing.

Owner: Phase 2 Implementation
Date: 2025-11-09
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.patches as mpatches


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix heatmap

    Args:
        confusion_matrix: 2x2 confusion matrix
        class_names: Class names
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add accuracy annotations
    TN, FP, FN, TP = confusion_matrix.ravel()
    total = TN + FP + FN + TP
    accuracy = (TN + TP) / total

    # Add text box with metrics
    textstr = f'Accuracy: {accuracy:.1%}\nTotal: {total} samples'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved confusion matrix to: {output_path}")


def visualize_misclassified_signals(
    errors: List[Dict],
    output_dir: Path,
    class_names: List[str] = None,
    max_examples_per_category: int = 5,
    max_grid_examples: int = 20
):
    """
    Generate visualizations for misclassified ECG signals

    Creates:
        - Grid view of top misclassified signals
        - Individual plots for each error category
        - Confidence distribution by error type

    Args:
        errors: List of error dictionaries with keys:
                - 'signal': ECG signal
                - 'true_label': True class
                - 'predicted_label': Predicted class
                - 'confidence': Prediction confidence
                - 'confidence_std': Confidence standard deviation
                - 'category': Error category
        output_dir: Output directory for plots
        class_names: Class names
        max_examples_per_category: Max individual plots per category
        max_grid_examples: Max signals in grid view
    """
    if class_names is None:
        class_names = ['Normal', 'Arrhythmia']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grid view of errors
    _plot_error_grid(errors[:max_grid_examples], output_dir, class_names)

    # 2. Individual error plots by category
    categories = set(e['category'] for e in errors)
    for category in categories:
        category_errors = [e for e in errors if e['category'] == category]
        _plot_category_errors(
            category_errors[:max_examples_per_category],
            category,
            output_dir,
            class_names
        )

    # 3. Confidence distributions by error type
    _plot_confidence_distributions(errors, output_dir)

    # 4. Error category summary
    _plot_error_category_summary(errors, output_dir)


def _plot_error_grid(
    errors: List[Dict],
    output_dir: Path,
    class_names: List[str]
):
    """Plot grid of misclassified signals"""
    n_errors = min(len(errors), 20)
    n_cols = 5
    n_rows = (n_errors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_errors > 1 else [axes]

    for i, error in enumerate(errors[:n_errors]):
        ax = axes[i]

        signal = error['signal']
        true_label = class_names[error['true_label']]
        pred_label = class_names[error['predicted_label']]
        confidence = error['confidence']
        category = error['category']

        # Plot signal
        ax.plot(signal, linewidth=0.5, color='red', alpha=0.7)
        ax.set_title(
            f"{category.capitalize()}\nTrue: {true_label}, Pred: {pred_label}\n"
            f"Conf: {confidence:.1%}",
            fontsize=8
        )
        ax.set_xlabel('Sample', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_errors, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Misclassified Signals Grid (Top {n_errors})',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / 'error_grid_all.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved error grid to: {output_path}")


def _plot_category_errors(
    category_errors: List[Dict],
    category: str,
    output_dir: Path,
    class_names: List[str]
):
    """Plot individual errors for a specific category"""
    if not category_errors:
        return

    n_errors = len(category_errors)
    fig, axes = plt.subplots(n_errors, 1, figsize=(12, 3 * n_errors))

    if n_errors == 1:
        axes = [axes]

    for i, error in enumerate(category_errors):
        ax = axes[i]

        signal = error['signal']
        true_label = class_names[error['true_label']]
        pred_label = class_names[error['predicted_label']]
        confidence = error['confidence']
        confidence_std = error.get('confidence_std', 0.0)

        # Plot signal
        ax.plot(signal, linewidth=1.0, color='red', alpha=0.8)
        ax.axhline(y=signal.mean(), linestyle='--', color='gray', alpha=0.5, linewidth=0.8)

        ax.set_title(
            f"ERROR #{i+1}: True={true_label}, Predicted={pred_label}, "
            f"Confidence={confidence:.1%} ± {confidence_std:.1%}",
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Normalized Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"Mean: {signal.mean():.3f}\nStd: {signal.std():.3f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{category.capitalize()} Errors (n={n_errors})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'error_category_{category}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved {category} errors to: {output_path}")


def _plot_confidence_distributions(errors: List[Dict], output_dir: Path):
    """Plot confidence distributions by error category"""
    categories = sorted(set(e['category'] for e in errors))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        if i >= 4:
            break

        ax = axes[i]
        category_errors = [e for e in errors if e['category'] == category]
        confidences = [e['confidence'] for e in category_errors]

        # Histogram
        ax.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=np.mean(confidences), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(confidences):.2%}')
        ax.axvline(x=np.median(confidences), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(confidences):.2%}')

        ax.set_title(f'{category.capitalize()} Errors (n={len(category_errors)})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(categories), 4):
        axes[i].axis('off')

    plt.suptitle('Confidence Distribution by Error Category',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'confidence_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved confidence distributions to: {output_path}")


def _plot_error_category_summary(errors: List[Dict], output_dir: Path):
    """Plot summary of error categories"""
    categories = sorted(set(e['category'] for e in errors))
    category_counts = {cat: sum(1 for e in errors if e['category'] == cat)
                      for cat in categories}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = sns.color_palette('Set2', len(categories))
    bars = ax1.bar(categories, [category_counts[c] for c in categories],
                   color=colors, edgecolor='black', alpha=0.8)

    ax1.set_xlabel('Error Category', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Error Distribution by Category', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Pie chart
    ax2.pie([category_counts[c] for c in categories], labels=categories,
            autopct='%1.1f%%', colors=colors, startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title(f'Error Category Distribution (Total: {len(errors)})',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'error_category_summary.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved error category summary to: {output_path}")


def plot_noise_robustness(
    noise_results: Dict[str, Dict],
    output_path: Path
):
    """
    Plot accuracy degradation curve vs SNR

    Args:
        noise_results: Results from test_additive_noise_robustness()
        output_path: Output path for plot
    """
    # Extract SNR levels and accuracies
    snr_levels = []
    accuracies = []
    class_0_accs = []
    class_1_accs = []

    for key in sorted(noise_results.keys()):
        if key == 'clean':
            continue
        if 'dB' in key:
            snr = int(key.replace('dB', ''))
            snr_levels.append(snr)
            accuracies.append(noise_results[key]['accuracy'])
            class_0_accs.append(noise_results[key].get('class_0_accuracy', 0))
            class_1_accs.append(noise_results[key].get('class_1_accuracy', 0))

    # Add clean baseline
    if 'clean' in noise_results:
        snr_levels.insert(0, 40)  # Represent clean as 40dB
        accuracies.insert(0, noise_results['clean']['accuracy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy curves
    ax.plot(snr_levels, accuracies, marker='o', linewidth=2.5,
            markersize=8, label='Overall Accuracy', color='steelblue')

    if class_0_accs and class_1_accs:
        ax.plot(snr_levels[1:], class_0_accs, marker='s', linewidth=2,
                markersize=7, label='Normal Class', color='green', linestyle='--')
        ax.plot(snr_levels[1:], class_1_accs, marker='^', linewidth=2,
                markersize=7, label='Arrhythmia Class', color='red', linestyle='--')

    # Add clinical threshold line
    ax.axhline(y=0.85, color='red', linestyle=':', linewidth=2,
               label='Clinical Threshold (85%)', alpha=0.7)

    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Robustness to Additive Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add SNR quality annotations
    ax.axvspan(30, 40, alpha=0.1, color='green', label='High Quality')
    ax.axvspan(20, 30, alpha=0.1, color='yellow')
    ax.axvspan(10, 20, alpha=0.1, color='orange')
    ax.axvspan(0, 10, alpha=0.1, color='red')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved noise robustness plot to: {output_path}")


def plot_signal_quality_comparison(
    quality_results: Dict[str, Dict],
    output_path: Path
):
    """
    Plot accuracy comparison across signal quality degradations

    Args:
        quality_results: Results from test_signal_quality_variations()
        output_path: Output path
    """
    degradations = list(quality_results.keys())
    accuracies = [quality_results[d]['accuracy'] for d in degradations]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette('Set2', len(degradations))
    bars = ax.bar(degradations, accuracies, color=colors,
                  edgecolor='black', alpha=0.8)

    # Add clinical threshold
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2,
               label='Clinical Threshold (85%)')

    ax.set_xlabel('Signal Quality Degradation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Under Signal Quality Variations',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved signal quality comparison to: {output_path}")


def plot_latency_distribution(
    latency_stats: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    Plot latency distribution box plots for different ensemble sizes

    Args:
        latency_stats: Dictionary with latency statistics
        output_path: Output path
    """
    ensemble_sizes = sorted(latency_stats.keys())
    data_to_plot = []
    labels = []

    for size in ensemble_sizes:
        stats = latency_stats[size]
        # Create synthetic data from percentiles (approximation)
        data = [
            stats['min'],
            stats['p95'] * 0.25 + stats['min'] * 0.75,  # Approximate Q1
            stats['median'],
            stats['p95'] * 0.75 + stats['median'] * 0.25,  # Approximate Q3
            stats['p95'],
            stats['p99'],
            stats['max']
        ]
        data_to_plot.append(data)
        labels.append(f"N={size}" if size != 'single' else "Single")

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     notch=True, showfliers=True)

    # Color boxes
    colors = sns.color_palette('pastel', len(ensemble_sizes))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add 50ms target line (real-time requirement)
    ax.axhline(y=50, color='green', linestyle='--', linewidth=2,
               label='Real-time Target (50ms)', alpha=0.7)

    ax.set_xlabel('Ensemble Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Latency Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved latency distribution to: {output_path}")


def plot_throughput_comparison(
    throughput_data: Dict[int, float],
    output_path: Path
):
    """
    Plot throughput vs batch size

    Args:
        throughput_data: {batch_size: samples_per_second}
        output_path: Output path
    """
    batch_sizes = sorted(throughput_data.keys())
    throughputs = [throughput_data[bs] for bs in batch_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(batch_sizes, throughputs, marker='o', linewidth=2.5,
            markersize=10, color='steelblue')

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Find optimal batch size
    optimal_bs = batch_sizes[np.argmax(throughputs)]
    optimal_throughput = max(throughputs)
    ax.axvline(x=optimal_bs, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {optimal_bs} (throughput={optimal_throughput:.1f}/s)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved throughput comparison to: {output_path}")
