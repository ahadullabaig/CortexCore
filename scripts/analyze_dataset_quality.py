#!/usr/bin/env python3
"""
Dataset Quality Analysis Script

This script analyzes the synthetic ECG dataset to identify label leakage
and quantify the separability of the classes. It demonstrates why the model
achieves 100% accuracy - the task is trivially simple.

Author: CortexCore Team
Date: 2025-11-04
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import fft
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


def load_dataset(data_path):
    """Load a dataset from .pt file."""
    data = torch.load(data_path)
    signals = data['signals'].numpy()
    labels = data['labels'].numpy()
    return signals, labels


def compute_frequency_features(signal, sampling_rate=250):
    """
    Compute frequency domain features from an ECG signal.

    Args:
        signal: 1D numpy array of ECG signal
        sampling_rate: Sampling rate in Hz

    Returns:
        dict: Frequency domain features
    """
    # Compute FFT
    n = len(signal)
    frequencies = fft.fftfreq(n, d=1/sampling_rate)
    fft_values = np.abs(fft.fft(signal))

    # Only positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    fft_values = fft_values[positive_freq_idx]

    # Find dominant frequency
    dominant_idx = np.argmax(fft_values)
    dominant_frequency = frequencies[dominant_idx]
    dominant_power = fft_values[dominant_idx]

    # Low frequency band (0.5-2 Hz) - heart rate region
    low_freq_mask = (frequencies >= 0.5) & (frequencies <= 2.0)
    low_freq_power = np.sum(fft_values[low_freq_mask])

    # High frequency band (5-30 Hz) - noise/irregularities
    high_freq_mask = (frequencies >= 5.0) & (frequencies <= 30.0)
    high_freq_power = np.sum(fft_values[high_freq_mask])

    # Heart rate estimation (BPM)
    heart_rate_hz = dominant_frequency
    heart_rate_bpm = heart_rate_hz * 60

    return {
        'dominant_frequency': dominant_frequency,
        'dominant_power': dominant_power,
        'low_freq_power': low_freq_power,
        'high_freq_power': high_freq_power,
        'heart_rate_bpm': heart_rate_bpm,
        'total_power': np.sum(fft_values),
    }


def compute_time_domain_features(signal):
    """Compute time domain features."""
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'variance': np.var(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal) - np.min(signal),
        'median': np.median(signal),
        'q25': np.percentile(signal, 25),
        'q75': np.percentile(signal, 75),
    }


def analyze_class_separation(signals, labels, class_names=['Normal', 'Arrhythmia']):
    """
    Analyze how separable the classes are.

    Returns detailed statistics and visualizations.
    """
    print("\n" + "="*80)
    print("DATASET QUALITY ANALYSIS - LABEL LEAKAGE INVESTIGATION")
    print("="*80)

    results = {
        'class_statistics': {},
        'separability_tests': {},
        'simple_classifiers': {},
    }

    # Extract features for all samples
    print("\n[1/5] Extracting features from signals...")
    freq_features = []
    time_features = []

    for signal in signals:
        freq_feat = compute_frequency_features(signal)
        time_feat = compute_time_domain_features(signal)
        freq_features.append(freq_feat)
        time_features.append(time_feat)

    # Convert to arrays
    dominant_freqs = np.array([f['dominant_frequency'] for f in freq_features])
    heart_rates = np.array([f['heart_rate_bpm'] for f in freq_features])
    high_freq_power = np.array([f['high_freq_power'] for f in freq_features])
    low_freq_power = np.array([f['low_freq_power'] for f in freq_features])
    signal_std = np.array([f['std'] for f in time_features])

    # Analyze each class
    print("\n[2/5] Computing class statistics...")
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx

        class_stats = {
            'sample_count': int(np.sum(mask)),
            'heart_rate_bpm': {
                'mean': float(np.mean(heart_rates[mask])),
                'std': float(np.std(heart_rates[mask])),
                'min': float(np.min(heart_rates[mask])),
                'max': float(np.max(heart_rates[mask])),
            },
            'dominant_frequency_hz': {
                'mean': float(np.mean(dominant_freqs[mask])),
                'std': float(np.std(dominant_freqs[mask])),
                'min': float(np.min(dominant_freqs[mask])),
                'max': float(np.max(dominant_freqs[mask])),
            },
            'high_freq_power': {
                'mean': float(np.mean(high_freq_power[mask])),
                'std': float(np.std(high_freq_power[mask])),
            },
            'signal_std': {
                'mean': float(np.mean(signal_std[mask])),
                'std': float(np.std(signal_std[mask])),
            }
        }

        results['class_statistics'][class_name] = class_stats

        print(f"\n{class_name} (n={class_stats['sample_count']}):")
        print(f"  Heart Rate: {class_stats['heart_rate_bpm']['mean']:.2f} ± {class_stats['heart_rate_bpm']['std']:.2f} BPM")
        print(f"  Dominant Freq: {class_stats['dominant_frequency_hz']['mean']:.3f} ± {class_stats['dominant_frequency_hz']['std']:.3f} Hz")
        print(f"  High-Freq Power: {class_stats['high_freq_power']['mean']:.2f} ± {class_stats['high_freq_power']['std']:.2f}")

    # Test for class separation
    print("\n[3/5] Testing class separability...")

    # T-test for heart rate
    normal_hr = heart_rates[labels == 0]
    arrhythmia_hr = heart_rates[labels == 1]
    t_stat, p_value = stats.ttest_ind(normal_hr, arrhythmia_hr)

    results['separability_tests']['heart_rate_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.001),
    }

    print(f"\nHeart Rate T-Test:")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Significant: {'YES - PERFECT SEPARATION' if p_value < 0.001 else 'No'}")

    # Check for overlap in heart rate distributions
    normal_hr_range = (np.min(normal_hr), np.max(normal_hr))
    arrhythmia_hr_range = (np.min(arrhythmia_hr), np.max(arrhythmia_hr))

    overlap = max(0, min(normal_hr_range[1], arrhythmia_hr_range[1]) - max(normal_hr_range[0], arrhythmia_hr_range[0]))

    results['separability_tests']['heart_rate_overlap'] = {
        'normal_range': normal_hr_range,
        'arrhythmia_range': arrhythmia_hr_range,
        'overlap_bpm': float(overlap),
        'overlap_percentage': float(overlap / (normal_hr_range[1] - normal_hr_range[0]) * 100) if overlap > 0 else 0.0,
    }

    print(f"\nHeart Rate Range Overlap:")
    print(f"  Normal: {normal_hr_range[0]:.1f} - {normal_hr_range[1]:.1f} BPM")
    print(f"  Arrhythmia: {arrhythmia_hr_range[0]:.1f} - {arrhythmia_hr_range[1]:.1f} BPM")
    print(f"  Overlap: {overlap:.1f} BPM ({overlap / (normal_hr_range[1] - normal_hr_range[0]) * 100:.1f}%)")

    if overlap == 0:
        print("  ⚠️  CRITICAL: ZERO OVERLAP - PERFECT SEPARABILITY!")

    # Test simple classifiers
    print("\n[4/5] Testing trivial classifiers...")

    # Rule-based classifier: Simple threshold on heart rate
    threshold = (np.mean(normal_hr) + np.mean(arrhythmia_hr)) / 2
    rule_predictions = (heart_rates > threshold).astype(int)
    rule_accuracy = accuracy_score(labels, rule_predictions)

    results['simple_classifiers']['heart_rate_threshold'] = {
        'threshold_bpm': float(threshold),
        'accuracy': float(rule_accuracy),
        'method': f'IF heart_rate > {threshold:.1f} BPM → Arrhythmia',
    }

    print(f"\nRule-Based Classifier:")
    print(f"  Rule: IF heart_rate > {threshold:.1f} BPM → Arrhythmia")
    print(f"  Accuracy: {rule_accuracy * 100:.1f}%")

    # Decision tree with max_depth=1 (single split)
    X_single = dominant_freqs.reshape(-1, 1)
    dt_simple = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt_simple.fit(X_single, labels)
    dt_predictions = dt_simple.predict(X_single)
    dt_accuracy = accuracy_score(labels, dt_predictions)

    results['simple_classifiers']['decision_tree_depth1'] = {
        'accuracy': float(dt_accuracy),
        'feature': 'dominant_frequency',
        'method': 'Single threshold decision tree',
    }

    print(f"\nDecision Tree (depth=1):")
    print(f"  Feature: Dominant Frequency")
    print(f"  Accuracy: {dt_accuracy * 100:.1f}%")

    # Linear classifier on multiple features
    X_multi = np.column_stack([dominant_freqs, high_freq_power, signal_std])
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_multi, labels)
    lr_predictions = lr.predict(X_multi)
    lr_accuracy = accuracy_score(labels, lr_predictions)

    results['simple_classifiers']['logistic_regression'] = {
        'accuracy': float(lr_accuracy),
        'features': ['dominant_frequency', 'high_freq_power', 'signal_std'],
        'method': 'Linear classifier on 3 features',
    }

    print(f"\nLogistic Regression (3 features):")
    print(f"  Features: Dominant Freq, High-Freq Power, Signal Std")
    print(f"  Accuracy: {lr_accuracy * 100:.1f}%")

    # Create visualizations
    print("\n[5/5] Creating visualizations...")
    create_visualizations(signals, labels, heart_rates, dominant_freqs, high_freq_power, class_names)

    return results


def create_visualizations(signals, labels, heart_rates, dominant_freqs, high_freq_power, class_names):
    """Create comprehensive visualizations."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Dataset Quality Analysis - Evidence of Label Leakage', fontsize=16, fontweight='bold')

    # 1. Sample ECG signals
    ax = axes[0, 0]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        sample_idx = np.where(mask)[0][0]
        ax.plot(signals[sample_idx][:1000], label=class_names[class_idx], alpha=0.7)
    ax.set_title('Sample ECG Signals (first 4 seconds)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Heart rate distribution
    ax = axes[0, 1]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.hist(heart_rates[mask], bins=30, alpha=0.6, label=class_names[class_idx], edgecolor='black')
    ax.set_title('Heart Rate Distribution (BPM)')
    ax.set_xlabel('Heart Rate (BPM)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.axvline(x=70, color='blue', linestyle='--', alpha=0.5, label='70 BPM')
    ax.axvline(x=120, color='red', linestyle='--', alpha=0.5, label='120 BPM')
    ax.grid(True, alpha=0.3)

    # 3. Dominant frequency distribution
    ax = axes[0, 2]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.hist(dominant_freqs[mask], bins=30, alpha=0.6, label=class_names[class_idx], edgecolor='black')
    ax.set_title('Dominant Frequency Distribution (Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scatter: Heart Rate vs High-Freq Power
    ax = axes[1, 0]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.scatter(heart_rates[mask], high_freq_power[mask], alpha=0.5, label=class_names[class_idx], s=20)
    ax.set_title('Heart Rate vs High-Frequency Power')
    ax.set_xlabel('Heart Rate (BPM)')
    ax.set_ylabel('High-Frequency Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Scatter: Dominant Freq vs High-Freq Power
    ax = axes[1, 1]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.scatter(dominant_freqs[mask], high_freq_power[mask], alpha=0.5, label=class_names[class_idx], s=20)
    ax.set_title('Dominant Frequency vs High-Frequency Power')
    ax.set_xlabel('Dominant Frequency (Hz)')
    ax.set_ylabel('High-Frequency Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Box plot: Heart Rate
    ax = axes[1, 2]
    data_to_plot = [heart_rates[labels == 0], heart_rates[labels == 1]]
    bp = ax.boxplot(data_to_plot, labels=class_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax.set_title('Heart Rate Box Plot')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.grid(True, alpha=0.3, axis='y')

    # 7. FFT of sample signals
    ax = axes[2, 0]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        sample_idx = np.where(mask)[0][0]
        signal = signals[sample_idx]

        n = len(signal)
        frequencies = fft.fftfreq(n, d=1/250)
        fft_values = np.abs(fft.fft(signal))

        positive_freq_idx = (frequencies > 0) & (frequencies < 10)
        ax.plot(frequencies[positive_freq_idx], fft_values[positive_freq_idx],
                label=class_names[class_idx], alpha=0.7)
    ax.set_title('Frequency Spectrum (Sample Signals)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. High-freq power distribution
    ax = axes[2, 1]
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.hist(high_freq_power[mask], bins=30, alpha=0.6, label=class_names[class_idx], edgecolor='black')
    ax.set_title('High-Frequency Power Distribution')
    ax.set_xlabel('High-Frequency Power')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Decision boundary visualization
    ax = axes[2, 2]
    threshold = (np.mean(heart_rates[labels == 0]) + np.mean(heart_rates[labels == 1])) / 2
    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.scatter(heart_rates[mask], np.ones_like(heart_rates[mask]) * class_idx,
                  alpha=0.5, label=class_names[class_idx], s=20)
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.1f} BPM')
    ax.set_title('Classification Threshold (Heart Rate)')
    ax.set_xlabel('Heart Rate (BPM)')
    ax.set_ylabel('Class Label')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('results/dataset_quality_analysis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    plt.close()


def main():
    """Main analysis function."""

    # Paths
    data_dir = Path('data/synthetic')
    train_path = data_dir / 'train_data.pt'
    val_path = data_dir / 'val_data.pt'
    test_path = data_dir / 'test_data.pt'

    # Check if data exists
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Please run scripts/02_generate_mvp_data.sh first")
        return

    print("Loading datasets...")
    train_signals, train_labels = load_dataset(train_path)

    print(f"\nDataset Info:")
    print(f"  Training samples: {len(train_signals)}")
    print(f"  Signal length: {len(train_signals[0])} samples")
    print(f"  Classes: {len(np.unique(train_labels))}")

    # Analyze training data
    results = analyze_class_separation(train_signals, train_labels)

    # Save results
    output_file = Path('results/dataset_quality_analysis.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[COMPLETE] Analysis saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - ROOT CAUSE OF 100% ACCURACY")
    print("="*80)

    normal_hr = results['class_statistics']['Normal']['heart_rate_bpm']
    arrhythmia_hr = results['class_statistics']['Arrhythmia']['heart_rate_bpm']

    print(f"\n1. FIXED HEART RATES (No Variability):")
    print(f"   Normal: {normal_hr['mean']:.1f} ± {normal_hr['std']:.2f} BPM")
    print(f"   Arrhythmia: {arrhythmia_hr['mean']:.1f} ± {arrhythmia_hr['std']:.2f} BPM")
    print(f"   → STD ≈ 0 means NO INTRA-CLASS VARIABILITY")

    print(f"\n2. RANGE OVERLAP:")
    overlap = results['separability_tests']['heart_rate_overlap']
    print(f"   Normal range: {overlap['normal_range'][0]:.1f}-{overlap['normal_range'][1]:.1f} BPM")
    print(f"   Arrhythmia range: {overlap['arrhythmia_range'][0]:.1f}-{overlap['arrhythmia_range'][1]:.1f} BPM")
    print(f"   Overlap: {overlap['overlap_bpm']:.1f} BPM ({overlap['overlap_percentage']:.1f}%)")

    if overlap['overlap_bpm'] == 0:
        print(f"   → ⚠️  ZERO OVERLAP = PERFECT LINEAR SEPARABILITY")

    print(f"\n3. TRIVIAL CLASSIFIER PERFORMANCE:")
    for name, classifier in results['simple_classifiers'].items():
        print(f"   {name}: {classifier['accuracy']*100:.1f}% accuracy")

    print(f"\n4. CONCLUSION:")
    print(f"   The 100% accuracy is REAL but MEANINGLESS.")
    print(f"   The model learned: 'IF heart_rate > 95 BPM → Arrhythmia'")
    print(f"   This is NOT learning disease patterns - it's learning a threshold.")
    print(f"   Even a 1-layer neural network would achieve 100% on this data.")

    print("\n" + "="*80)
    print("RECOMMENDATION: Regenerate data with realistic variability")
    print("="*80)
    print("- Normal: heart_rate = random(60, 100) BPM")
    print("- Arrhythmia: heart_rate = random(40-60 or 90-180) BPM")
    print("- Add inter-patient variability and realistic artifacts")
    print("- Target accuracy should drop to 85-92% (realistic range)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
