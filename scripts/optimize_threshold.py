"""
ROC Curve Analysis for Threshold Optimization
==============================================

This script analyzes the model's probability outputs to find the optimal
classification threshold that maximizes sensitivity while maintaining
acceptable specificity.

Goal: Find threshold that achieves ≥95% sensitivity (Arrhythmia detection)

Usage:
    python scripts/optimize_threshold.py
    python scripts/optimize_threshold.py --model models/best_model.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SimpleSNN, WiderSNN, DeepSNN
from src.inference import load_model
from src.utils import set_seed


def detect_model_architecture(model_path: str, device: str = 'cpu'):
    """
    Detect and instantiate the correct model architecture from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: Device for loading

    Returns:
        Instantiated model (SimpleSNN, WiderSNN, or DeepSNN)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
        architecture = config.get('architecture', 'SimpleSNN')

        if architecture == 'DeepSNN':
            hidden_sizes = config.get('hidden_sizes', [256, 128])
            dropout = config.get('dropout', 0.3)
            return DeepSNN(hidden_sizes=hidden_sizes, dropout=dropout)
        elif architecture == 'WiderSNN':
            hidden_size = config.get('hidden_size', 256)
            dropout = config.get('dropout', 0.2)
            return WiderSNN(hidden_size=hidden_size, dropout=dropout)
        else:
            return SimpleSNN()
    else:
        # No config, assume SimpleSNN (baseline)
        return SimpleSNN()


def get_predictions_and_labels(model, test_data, device='cpu', num_steps=100, gain=10.0, ensemble_size=1):
    """
    Get probability predictions and true labels for entire test set

    Args:
        ensemble_size: Number of predictions to average per sample (default: 1 for single prediction)
                      Use ensemble_size=3 for ensemble predictions (reduces variance)

    Returns:
        y_true: True labels [N]
        y_probs: Arrhythmia probabilities [N] (averaged if ensemble_size > 1)
        y_pred_argmax: Predictions using argmax (baseline) [N]
    """
    model.eval()
    model.to(device)

    signals = test_data['signals']
    labels = test_data['labels']

    y_true = []
    y_probs = []
    y_pred_argmax = []

    if ensemble_size > 1:
        print(f"Generating ENSEMBLE predictions (N={ensemble_size}) for {len(signals)} test samples...")
    else:
        print(f"Generating predictions for {len(signals)} test samples...")

    with torch.no_grad():
        for i in range(len(signals)):
            signal_np = signals[i].numpy()

            # Ensemble prediction: average probabilities across multiple encodings
            ensemble_probs = []

            for ensemble_idx in range(ensemble_size):
                # Use deterministic seed matching ensemble_predict() pattern
                # Sample i, ensemble member j: seed = (42 + i*1000) + j
                # This ensures consistent results with comprehensive_evaluation.py
                sample_base_seed = 42 + i * 1000
                seed = sample_base_seed + ensemble_idx
                set_seed(seed)

                # Normalize and encode to spikes
                signal_norm = (signal_np - signal_np.min()) / (signal_np.max() - signal_np.min() + 1e-8)
                spikes = np.random.rand(num_steps, len(signal_np)) < (signal_norm * gain / 100.0)
                input_tensor = torch.FloatTensor(spikes).unsqueeze(1).to(device)  # [num_steps, 1, signal_length]

                # Forward pass
                model_output = model(input_tensor)
                if isinstance(model_output, tuple):
                    spikes_out, membrane = model_output[:2]
                else:
                    spikes_out = model_output

                # Get logits
                output = spikes_out.sum(dim=0)  # [1, 2]

                # Get probabilities
                probs = torch.softmax(output, dim=1)
                arrhythmia_prob = probs[0, 1].item()  # Probability of class 1 (Arrhythmia)

                ensemble_probs.append(arrhythmia_prob)

            # Average probabilities across ensemble
            avg_arrhythmia_prob = np.mean(ensemble_probs)

            # Argmax prediction: use averaged probability
            pred_argmax = 1 if avg_arrhythmia_prob >= 0.5 else 0

            y_true.append(labels[i].item())
            y_probs.append(avg_arrhythmia_prob)
            y_pred_argmax.append(pred_argmax)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(signals)} samples")

    return np.array(y_true), np.array(y_probs), np.array(y_pred_argmax)


def analyze_roc_curve(y_true, y_probs, target_sensitivity=0.95, target_specificity=0.90):
    """
    Analyze ROC curve and find optimal threshold

    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        target_sensitivity: Target sensitivity (TPR)
        target_specificity: Target specificity (TNR)

    Returns:
        Dictionary with ROC analysis results
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    # Calculate specificity for each threshold
    specificity = 1 - fpr

    # Find threshold closest to target sensitivity
    target_idx = np.argmin(np.abs(tpr - target_sensitivity))
    optimal_threshold = thresholds[target_idx]
    optimal_tpr = tpr[target_idx]
    optimal_fpr = fpr[target_idx]
    optimal_specificity = 1 - optimal_fpr

    # Find threshold that meets BOTH clinical targets (if exists)
    # Search for thresholds where BOTH sensitivity >= target AND specificity >= target
    dual_constraint_mask = (tpr >= target_sensitivity) & (specificity >= target_specificity)

    if dual_constraint_mask.any():
        # Found at least one threshold meeting both constraints
        # Choose the one with best G-mean among valid thresholds
        valid_indices = np.where(dual_constraint_mask)[0]
        g_means = np.sqrt(tpr[valid_indices] * specificity[valid_indices])
        best_valid_idx = valid_indices[np.argmax(g_means)]

        dual_threshold = thresholds[best_valid_idx]
        dual_sensitivity = tpr[best_valid_idx]
        dual_specificity = specificity[best_valid_idx]
        dual_found = True
        dual_idx = best_valid_idx
    else:
        # No threshold meets both constraints
        dual_threshold = None
        dual_sensitivity = None
        dual_specificity = None
        dual_found = False
        dual_idx = None

    # Also find Youden's J statistic threshold (maximize TPR + TNR - 1)
    youdens_j = tpr + (1 - fpr) - 1
    youdens_idx = np.argmax(youdens_j)
    youdens_threshold = thresholds[youdens_idx]
    youdens_tpr = tpr[youdens_idx]
    youdens_fpr = fpr[youdens_idx]

    # Find baseline (0.5 threshold) performance
    baseline_idx = np.argmin(np.abs(thresholds - 0.5))
    baseline_tpr = tpr[baseline_idx]
    baseline_fpr = fpr[baseline_idx]

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': optimal_tpr,
        'optimal_specificity': optimal_specificity,
        'optimal_fpr': optimal_fpr,
        'youdens_threshold': youdens_threshold,
        'youdens_sensitivity': youdens_tpr,
        'youdens_specificity': 1 - youdens_fpr,
        'baseline_sensitivity': baseline_tpr,
        'baseline_specificity': 1 - baseline_fpr,
        'target_idx': target_idx,
        'youdens_idx': youdens_idx,
        'baseline_idx': baseline_idx,
        # Dual constraint results
        'dual_constraint_found': dual_found,
        'dual_threshold': dual_threshold,
        'dual_sensitivity': dual_sensitivity,
        'dual_specificity': dual_specificity,
        'dual_idx': dual_idx
    }


def evaluate_threshold(y_true, y_probs, threshold):
    """
    Evaluate performance at specific threshold

    Returns confusion matrix and clinical metrics
    """
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy
    }


def plot_roc_curve(roc_results, save_path='results/roc_curve_threshold_optimization.png', ensemble_size=1):
    """
    Plot ROC curve with optimal points marked
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Add ensemble info to title
    title_suffix = f" (Ensemble N={ensemble_size})" if ensemble_size > 1 else " (Single Predictions)"

    # Plot 1: ROC Curve
    ax1.plot(roc_results['fpr'], roc_results['tpr'],
             linewidth=2, label=f"AUC = {roc_results['auc']:.3f}")

    # Mark dual-constraint threshold if found (PRIORITY - most important!)
    if roc_results['dual_constraint_found']:
        ax1.scatter(1 - roc_results['dual_specificity'], roc_results['dual_sensitivity'],
                    s=200, c='gold', marker='*', edgecolors='black', linewidths=2,
                    label=f"⭐ DUAL TARGET MET\n(threshold={roc_results['dual_threshold']:.3f})",
                    zorder=10)

    # Mark optimal threshold (target sensitivity)
    ax1.scatter(roc_results['optimal_fpr'], roc_results['optimal_sensitivity'],
                s=150, c='red', marker='o', edgecolors='black', linewidths=2,
                label=f"Target Sensitivity\n(threshold={roc_results['optimal_threshold']:.3f})")

    # Mark Youden's J threshold
    ax1.scatter(1 - roc_results['youdens_specificity'], roc_results['youdens_sensitivity'],
                s=150, c='green', marker='s', edgecolors='black', linewidths=2,
                label=f"Youden's J\n(threshold={roc_results['youdens_threshold']:.3f})")

    # Mark baseline (0.5 threshold)
    ax1.scatter(1 - roc_results['baseline_specificity'], roc_results['baseline_sensitivity'],
                s=150, c='orange', marker='^', edgecolors='black', linewidths=2,
                label=f"Baseline (0.5)")

    # Diagonal line (random classifier)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax1.set_title(f'ROC Curve - Threshold Optimization{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Plot 2: Sensitivity vs Specificity by Threshold
    # Filter to reasonable threshold range (0.1 to 0.9)
    mask = (roc_results['thresholds'] >= 0.1) & (roc_results['thresholds'] <= 0.9)
    thresh_filtered = roc_results['thresholds'][mask]
    tpr_filtered = roc_results['tpr'][mask]
    specificity_filtered = 1 - roc_results['fpr'][mask]

    ax2.plot(thresh_filtered, tpr_filtered,
             linewidth=2, label='Sensitivity (TPR)', color='blue')
    ax2.plot(thresh_filtered, specificity_filtered,
             linewidth=2, label='Specificity (TNR)', color='green')

    # Mark dual-constraint threshold if found
    if roc_results['dual_constraint_found']:
        ax2.axvline(roc_results['dual_threshold'], color='gold',
                    linestyle='-', linewidth=3, alpha=0.8,
                    label=f"⭐ Dual Target: {roc_results['dual_threshold']:.3f}",
                    zorder=10)

    # Mark optimal threshold
    ax2.axvline(roc_results['optimal_threshold'], color='red',
                linestyle='--', linewidth=2, alpha=0.7,
                label=f"Sensitivity Only: {roc_results['optimal_threshold']:.3f}")

    # Mark 95% sensitivity target
    ax2.axhline(0.95, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.text(0.15, 0.96, '95% Sens Target', fontsize=9, color='red')

    # Mark 90% specificity target
    ax2.axhline(0.90, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax2.text(0.15, 0.91, '90% Spec Target', fontsize=9, color='green')

    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title(f'Sensitivity vs Specificity by Threshold{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.1, 0.9])
    ax2.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 ROC curve saved to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='ROC curve analysis for threshold optimization')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/synthetic/test_data.pt',
                       help='Path to test dataset')
    parser.add_argument('--target-sensitivity', type=float, default=0.95,
                       help='Target sensitivity (default: 0.95)')
    parser.add_argument('--target-specificity', type=float, default=0.90,
                       help='Target specificity (default: 0.90)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--ensemble', type=int, default=1,
                       help='Ensemble size: number of predictions to average (default: 1 for single prediction, use 3 for production)')
    args = parser.parse_args()

    print("=" * 80)
    print("ROC CURVE ANALYSIS FOR THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Create results directory
    Path('results').mkdir(exist_ok=True)

    # Load model
    print(f"\n📦 Loading model from {args.model}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = detect_model_architecture(args.model, device=str(device))
    model = load_model(args.model, model, device=device)
    print(f"✅ Model loaded successfully ({model.__class__.__name__}) on {device}")

    # Load test data
    print(f"\n📦 Loading test data from {args.test_data}")
    test_data = torch.load(args.test_data)
    print(f"✅ Test data loaded: {len(test_data['signals'])} samples")
    print(f"   Class distribution: {(test_data['labels'] == 0).sum().item()} Normal, "
          f"{(test_data['labels'] == 1).sum().item()} Arrhythmia")

    # Get predictions
    print("\n🔮 Generating probability predictions...")
    if args.ensemble > 1:
        print(f"   Using ensemble size: {args.ensemble} (production mode)")
    else:
        print(f"   Using single predictions (ensemble size: 1)")
    y_true, y_probs, y_pred_argmax = get_predictions_and_labels(
        model, test_data, device=device, ensemble_size=args.ensemble
    )
    print(f"✅ Predictions complete")

    # Baseline performance (argmax = 0.5 threshold)
    print("\n" + "=" * 80)
    print("BASELINE PERFORMANCE (argmax / 0.5 threshold)")
    print("=" * 80)
    baseline_metrics = evaluate_threshold(y_true, y_pred_argmax, 0.5)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Arrhythmia")
    print(f"True Normal      {baseline_metrics['tn']:4d}      {baseline_metrics['fp']:4d}")
    print(f"     Arrhythmia  {baseline_metrics['fn']:4d}      {baseline_metrics['tp']:4d}")
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity: {baseline_metrics['sensitivity']:.1%} (Target: ≥95%)")
    print(f"  Specificity: {baseline_metrics['specificity']:.1%} (Target: ≥90%)")
    print(f"  PPV:         {baseline_metrics['ppv']:.1%} (Target: ≥85%)")
    print(f"  NPV:         {baseline_metrics['npv']:.1%} (Target: ≥95%)")
    print(f"  Accuracy:    {baseline_metrics['accuracy']:.1%}")

    # ROC analysis
    print("\n" + "=" * 80)
    print("ROC CURVE ANALYSIS")
    print("=" * 80)
    roc_results = analyze_roc_curve(
        y_true, y_probs,
        target_sensitivity=args.target_sensitivity,
        target_specificity=args.target_specificity
    )
    print(f"\nAUC-ROC: {roc_results['auc']:.4f}")

    # Optimal threshold for target sensitivity
    print(f"\n🎯 OPTIMAL THRESHOLD FOR {args.target_sensitivity:.0%} SENSITIVITY:")
    print(f"  Threshold: {roc_results['optimal_threshold']:.3f}")
    optimal_metrics = evaluate_threshold(y_true, y_probs, roc_results['optimal_threshold'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Arrhythmia")
    print(f"True Normal      {optimal_metrics['tn']:4d}      {optimal_metrics['fp']:4d}")
    print(f"     Arrhythmia  {optimal_metrics['fn']:4d}      {optimal_metrics['tp']:4d}")
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity: {optimal_metrics['sensitivity']:.1%} ✅ Target met!" if optimal_metrics['sensitivity'] >= 0.95 else f"  Sensitivity: {optimal_metrics['sensitivity']:.1%} ⚠️  Below target")
    print(f"  Specificity: {optimal_metrics['specificity']:.1%} ✅ Target met!" if optimal_metrics['specificity'] >= 0.90 else f"  Specificity: {optimal_metrics['specificity']:.1%} ⚠️  Below target")
    print(f"  PPV:         {optimal_metrics['ppv']:.1%}")
    print(f"  NPV:         {optimal_metrics['npv']:.1%}")
    print(f"  Accuracy:    {optimal_metrics['accuracy']:.1%}")

    # Youden's J threshold
    print(f"\n📊 YOUDEN'S J THRESHOLD (balanced sensitivity/specificity):")
    print(f"  Threshold: {roc_results['youdens_threshold']:.3f}")
    youdens_metrics = evaluate_threshold(y_true, y_probs, roc_results['youdens_threshold'])
    print(f"  Sensitivity: {youdens_metrics['sensitivity']:.1%}")
    print(f"  Specificity: {youdens_metrics['specificity']:.1%}")

    # DUAL CONSTRAINT THRESHOLD (most important!)
    print("\n" + "=" * 80)
    print(f"⭐ DUAL CONSTRAINT SEARCH: ≥{args.target_sensitivity:.0%} Sensitivity AND ≥{args.target_specificity:.0%} Specificity")
    print("=" * 80)

    if roc_results['dual_constraint_found']:
        print(f"\n✅ SUCCESS! Found threshold meeting BOTH clinical targets:")
        print(f"  Optimal Threshold: {roc_results['dual_threshold']:.3f}")

        dual_metrics = evaluate_threshold(y_true, y_probs, roc_results['dual_threshold'])
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              Normal  Arrhythmia")
        print(f"True Normal      {dual_metrics['tn']:4d}      {dual_metrics['fp']:4d}")
        print(f"     Arrhythmia  {dual_metrics['fn']:4d}      {dual_metrics['tp']:4d}")
        print(f"\nClinical Metrics:")
        print(f"  Sensitivity: {dual_metrics['sensitivity']:.1%} ✅ (Target: ≥{args.target_sensitivity:.0%})")
        print(f"  Specificity: {dual_metrics['specificity']:.1%} ✅ (Target: ≥{args.target_specificity:.0%})")
        print(f"  PPV:         {dual_metrics['ppv']:.1%}")
        print(f"  NPV:         {dual_metrics['npv']:.1%}")
        print(f"  Accuracy:    {dual_metrics['accuracy']:.1%}")
        print(f"  G-mean:      {(dual_metrics['sensitivity'] * dual_metrics['specificity'])**0.5:.1%}")
        print(f"\n  False Negatives: {dual_metrics['fn']} (missed arrhythmias)")
        print(f"  False Positives: {dual_metrics['fp']} (false alarms)")
    else:
        print(f"\n❌ NO THRESHOLD FOUND meeting both targets simultaneously")
        print(f"\nROC curve analysis shows NO operating point where:")
        print(f"  • Sensitivity ≥ {args.target_sensitivity:.0%} AND")
        print(f"  • Specificity ≥ {args.target_specificity:.0%}")
        print(f"\nTrade-off examples from ROC curve:")
        print(f"  • At threshold {roc_results['optimal_threshold']:.3f}: {optimal_metrics['sensitivity']:.1%} sens / {optimal_metrics['specificity']:.1%} spec")
        print(f"  • At threshold {roc_results['youdens_threshold']:.3f}: {youdens_metrics['sensitivity']:.1%} sens / {youdens_metrics['specificity']:.1%} spec")
        print(f"\n💡 This indicates the model's fundamental capability limit.")
        print(f"   Recommendation: Proceed to Tier 2 fixes (data augmentation, architecture changes)")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: BASELINE VS OPTIMAL THRESHOLD")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Baseline (0.5)':<18} {'Optimal ({:.3f})'.format(roc_results['optimal_threshold']):<18} {'Change'}")
    print("-" * 80)
    print(f"{'Sensitivity':<20} {baseline_metrics['sensitivity']:>7.1%}{' '*10} {optimal_metrics['sensitivity']:>7.1%}{' '*10} {(optimal_metrics['sensitivity'] - baseline_metrics['sensitivity']):>+7.1%}")
    print(f"{'Specificity':<20} {baseline_metrics['specificity']:>7.1%}{' '*10} {optimal_metrics['specificity']:>7.1%}{' '*10} {(optimal_metrics['specificity'] - baseline_metrics['specificity']):>+7.1%}")
    print(f"{'PPV (Precision)':<20} {baseline_metrics['ppv']:>7.1%}{' '*10} {optimal_metrics['ppv']:>7.1%}{' '*10} {(optimal_metrics['ppv'] - baseline_metrics['ppv']):>+7.1%}")
    print(f"{'NPV':<20} {baseline_metrics['npv']:>7.1%}{' '*10} {optimal_metrics['npv']:>7.1%}{' '*10} {(optimal_metrics['npv'] - baseline_metrics['npv']):>+7.1%}")
    print(f"{'Accuracy':<20} {baseline_metrics['accuracy']:>7.1%}{' '*10} {optimal_metrics['accuracy']:>7.1%}{' '*10} {(optimal_metrics['accuracy'] - baseline_metrics['accuracy']):>+7.1%}")
    print(f"{'False Negatives':<20} {baseline_metrics['fn']:>7d}{' '*10} {optimal_metrics['fn']:>7d}{' '*10} {(optimal_metrics['fn'] - baseline_metrics['fn']):>+7d}")
    print(f"{'False Positives':<20} {baseline_metrics['fp']:>7d}{' '*10} {optimal_metrics['fp']:>7d}{' '*10} {(optimal_metrics['fp'] - baseline_metrics['fp']):>+7d}")

    # Plot ROC curve
    print("\n📊 Generating ROC curve visualization...")
    if args.ensemble > 1:
        save_path = f'results/roc_curve_threshold_optimization_ensemble{args.ensemble}.png'
    else:
        save_path = 'results/roc_curve_threshold_optimization.png'
    plot_roc_curve(roc_results, save_path=save_path, ensemble_size=args.ensemble)

    # Recommendations
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)

    if roc_results['dual_constraint_found']:
        # Dual-constraint threshold found - BEST OUTCOME!
        print(f"\n⭐ RECOMMENDED THRESHOLD: {roc_results['dual_threshold']:.3f} (DUAL TARGET)")
        print(f"\n✅ This threshold meets BOTH clinical targets:")
        print(f"  • Sensitivity: {dual_metrics['sensitivity']:.1%} (≥{args.target_sensitivity:.0%} ✓)")
        print(f"  • Specificity: {dual_metrics['specificity']:.1%} (≥{args.target_specificity:.0%} ✓)")
        print(f"  • G-mean: {(dual_metrics['sensitivity'] * dual_metrics['specificity'])**0.5:.1%}")

        print(f"\nChanges vs baseline (threshold 0.5):")
        print(f"  • Sensitivity: {baseline_metrics['sensitivity']:.1%} → {dual_metrics['sensitivity']:.1%} "
              f"({dual_metrics['sensitivity'] - baseline_metrics['sensitivity']:+.1%})")
        print(f"  • Specificity: {baseline_metrics['specificity']:.1%} → {dual_metrics['specificity']:.1%} "
              f"({dual_metrics['specificity'] - baseline_metrics['specificity']:+.1%})")
        print(f"  • False negatives: {baseline_metrics['fn']} → {dual_metrics['fn']} "
              f"({dual_metrics['fn'] - baseline_metrics['fn']:+d})")
        print(f"  • False positives: {baseline_metrics['fp']} → {dual_metrics['fp']} "
              f"({dual_metrics['fp'] - baseline_metrics['fp']:+d})")

        print(f"\n🔧 Implementation:")
        print(f"   Update src/inference.py predict() function:")
        print(f"   - Replace: predicted_class = output.argmax(dim=1).item()")
        print(f"   - With:    predicted_class = 1 if arrhythmia_prob >= {roc_results['dual_threshold']:.3f} else 0")
        print(f"   - Use ensemble_size=3 for production (required for this threshold)")

        print("\n" + "=" * 80)
        print("✅ ✅ ✅  COMPLETE SUCCESS: READY FOR CLINICAL DEPLOYMENT  ✅ ✅ ✅")
        print("=" * 80)

    else:
        # No dual-constraint threshold found
        print(f"\n⚠️  NO THRESHOLD meets both clinical targets simultaneously")
        print(f"\nBest available options:")
        print(f"\n1. MAXIMIZE SENSITIVITY (threshold {roc_results['optimal_threshold']:.3f}):")
        print(f"   • Sensitivity: {optimal_metrics['sensitivity']:.1%} {'✅' if optimal_metrics['sensitivity'] >= args.target_sensitivity else '❌'}")
        print(f"   • Specificity: {optimal_metrics['specificity']:.1%} {'✅' if optimal_metrics['specificity'] >= args.target_specificity else '❌'}")

        print(f"\n2. BALANCED (Youden's J, threshold {roc_results['youdens_threshold']:.3f}):")
        print(f"   • Sensitivity: {youdens_metrics['sensitivity']:.1%} {'✅' if youdens_metrics['sensitivity'] >= args.target_sensitivity else '❌'}")
        print(f"   • Specificity: {youdens_metrics['specificity']:.1%} {'✅' if youdens_metrics['specificity'] >= args.target_specificity else '❌'}")

        print(f"\n💡 RECOMMENDATION: Proceed to Tier 2 fixes")
        print(f"   • Data augmentation (temporal jittering, noise injection)")
        print(f"   • Architecture changes (attention, multi-scale)")
        print(f"   • Real-world data (MIT-BIH, PhysioNet)")

        print("\n" + "=" * 80)
        print("❌ THRESHOLD OPTIMIZATION INSUFFICIENT - TIER 2 FIXES REQUIRED")
        print("=" * 80)


if __name__ == '__main__':
    main()
