#!/usr/bin/env python3
"""
Comprehensive Model Evaluation - Phase 2
=========================================

Implements all 5 evaluation tasks from docs/NEXT_STEPS_DETAILED.md Phase 2:
    2.1: Full Test Set Analysis
    2.2: Clinical Performance Metrics
    2.3: Error Pattern Analysis
    2.4: Robustness Testing
    2.5: Performance Benchmarking

Owner: Phase 2 Implementation
Date: 2025-11-09
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import argparse
import json
import logging
import time
import psutil
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# Project imports
from src.model import SimpleSNN, WiderSNN, DeepSNN
from src.inference import load_model, ensemble_predict
from src.utils import set_seed, get_device
from src.evaluation import metrics, robustness, visualizations, reports


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"   ✅ Saved results to: {output_path}")


def detect_model_architecture(model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Detect and instantiate the correct model architecture from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: Device for loading

    Returns:
        Instantiated model (SimpleSNN, WiderSNN, or DeepSNN)
    """
    # Load checkpoint to read config
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


def run_task_2_1(
    model: nn.Module,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    ensemble_size: int,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Task 2.1: Full Test Set Analysis

    Run model on all 1000 test samples with ensemble predictions

    Returns:
        Dictionary with test set analysis results
    """
    logger.info("Running Task 2.1: Full Test Set Analysis...")

    n_samples = len(test_signals)
    logger.info(f"   Test samples: {n_samples}")
    logger.info(f"   Ensemble size: {ensemble_size}")

    predictions = []
    confidences = []
    confidence_stds = []
    agreement_rates = []
    inference_times = []
    probabilities_all = []

    # Run predictions with progress bar
    for i, signal in enumerate(tqdm(test_signals, desc="Evaluating test set")):
        # Use deterministic seed for reproducibility
        # Seed pattern matches optimize_threshold.py: 42 + sample_idx * 1000
        # This ensures each sample gets unique seeds for ensemble members:
        #   Sample 0: seeds 42, 43, 44 (for ensemble_size=3)
        #   Sample 1: seeds 1042, 1043, 1044
        #   etc.
        sample_base_seed = 42 + i * 1000

        result = ensemble_predict(
            model=model,
            input_data=signal,
            ensemble_size=ensemble_size,
            device=str(device),
            return_confidence=True,
            base_seed=sample_base_seed  # CRITICAL: Ensures deterministic spike encoding
        )

        predictions.append(result['prediction'])
        confidences.append(result['confidence'])
        confidence_stds.append(result.get('confidence_std', 0.0))
        agreement_rates.append(result.get('agreement_rate', 1.0))
        inference_times.append(result['inference_time_ms'])
        probabilities_all.append(result['probabilities'])

    # Convert to numpy
    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    # Calculate overall accuracy
    overall_accuracy = (predictions == test_labels).mean()

    # Per-class accuracy
    class_0_mask = test_labels == 0
    class_1_mask = test_labels == 1

    class_0_accuracy = (predictions[class_0_mask] == test_labels[class_0_mask]).mean()
    class_1_accuracy = (predictions[class_1_mask] == test_labels[class_1_mask]).mean()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm = sklearn_cm(test_labels, predictions)

    # Results
    results = {
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': {
            'Normal': float(class_0_accuracy),
            'Arrhythmia': float(class_1_accuracy)
        },
        'confusion_matrix': cm.tolist(),
        'total_samples': int(n_samples),
        'ensemble_size': int(ensemble_size),
        'mean_inference_time_ms': float(np.mean(inference_times)),
        'std_inference_time_ms': float(np.std(inference_times)),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'mean_agreement_rate': float(np.mean(agreement_rates)),
        'predictions': predictions.tolist(),
        'true_labels': test_labels.tolist(),
        'confidences': confidences,
        'confidence_stds': confidence_stds,
        'probabilities': probabilities_all
    }

    logger.info(f"   Overall Accuracy: {overall_accuracy:.1%}")
    logger.info(f"   Normal Accuracy: {class_0_accuracy:.1%}")
    logger.info(f"   Arrhythmia Accuracy: {class_1_accuracy:.1%}")
    logger.info(f"   Mean Inference Time: {np.mean(inference_times):.1f}ms")

    return results


def run_task_2_2(
    task_2_1_results: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Task 2.2: Clinical Performance Metrics

    Calculate comprehensive clinical metrics from test set results

    Args:
        task_2_1_results: Results from Task 2.1

    Returns:
        Dictionary with clinical metrics
    """
    logger.info("Running Task 2.2: Clinical Performance Metrics...")

    predictions = np.array(task_2_1_results['predictions'])
    true_labels = np.array(task_2_1_results['true_labels'])
    probabilities = np.array(task_2_1_results['probabilities'])

    # Calculate comprehensive clinical metrics
    clinical_metrics = metrics.calculate_comprehensive_clinical_metrics(
        y_true=true_labels,
        y_pred=predictions,
        y_proba=probabilities,
        class_names=['Normal', 'Arrhythmia']
    )

    # Calculate per-class metrics
    per_class_metrics = metrics.calculate_per_class_metrics(
        y_true=true_labels,
        y_pred=predictions,
        class_names=['Normal', 'Arrhythmia']
    )

    results = {
        **clinical_metrics,
        'per_class_metrics': per_class_metrics
    }

    # Log key metrics
    bin_metrics = clinical_metrics['binary_metrics']
    logger.info(f"   Sensitivity: {bin_metrics['sensitivity']:.1%}")
    logger.info(f"   Specificity: {bin_metrics['specificity']:.1%}")
    logger.info(f"   Precision: {bin_metrics['precision']:.1%}")
    logger.info(f"   NPV: {bin_metrics['npv']:.1%}")
    logger.info(f"   F1-Score: {bin_metrics['f1_score']:.3f}")

    # Log target status
    targets = clinical_metrics['targets']
    targets_met = sum([
        targets['sensitivity_met'],
        targets['specificity_met'],
        targets['precision_met'],
        targets['npv_met']
    ])
    logger.info(f"   Clinical Targets Met: {targets_met}/4")

    return results


def run_task_2_3(
    task_2_1_results: Dict[str, Any],
    test_signals: np.ndarray,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Task 2.3: Error Pattern Analysis

    Extract and categorize all misclassified samples

    Args:
        task_2_1_results: Results from Task 2.1
        test_signals: Test signals
        output_dir: Output directory for visualizations

    Returns:
        Dictionary with error analysis results
    """
    logger.info("Running Task 2.3: Error Pattern Analysis...")

    predictions = np.array(task_2_1_results['predictions'])
    true_labels = np.array(task_2_1_results['true_labels'])
    confidences = np.array(task_2_1_results['confidences'])
    confidence_stds = np.array(task_2_1_results['confidence_stds'])

    # Find misclassified samples
    error_mask = predictions != true_labels
    error_indices = np.where(error_mask)[0]

    logger.info(f"   Total errors: {len(error_indices)}")

    # Extract error information
    errors = []
    for idx in error_indices:
        signal = test_signals[idx]
        confidence = confidences[idx]
        confidence_std = confidence_stds[idx]

        # Categorize error
        if confidence < 0.6 and confidence_std > 0.15:
            category = 'borderline'  # Model uncertain
        elif confidence_std > 0.20:
            category = 'noisy'  # High variance = noisy
        elif signal.std() < 0.1:
            category = 'atypical'  # Unusual morphology
        else:
            category = 'systematic'  # Model consistently wrong

        errors.append({
            'index': int(idx),
            'true_label': int(true_labels[idx]),
            'predicted_label': int(predictions[idx]),
            'confidence': float(confidence),
            'confidence_std': float(confidence_std),
            'category': category,
            'signal': signal.tolist()
        })

    # Count by category
    error_categories = {}
    for category in ['borderline', 'noisy', 'atypical', 'systematic']:
        category_errors = [e for e in errors if e['category'] == category]
        error_categories[category] = {
            'count': len(category_errors),
            'mean_confidence': float(np.mean([e['confidence'] for e in category_errors])) if category_errors else 0.0,
            'std_confidence': float(np.std([e['confidence'] for e in category_errors])) if category_errors else 0.0
        }

    # Count false positives and false negatives
    false_positives = sum(1 for e in errors if e['true_label'] == 0 and e['predicted_label'] == 1)
    false_negatives = sum(1 for e in errors if e['true_label'] == 1 and e['predicted_label'] == 0)

    logger.info(f"   False Positives: {false_positives}")
    logger.info(f"   False Negatives: {false_negatives}")
    logger.info(f"   Borderline: {error_categories['borderline']['count']}")
    logger.info(f"   Noisy: {error_categories['noisy']['count']}")
    logger.info(f"   Atypical: {error_categories['atypical']['count']}")
    logger.info(f"   Systematic: {error_categories['systematic']['count']}")

    # Generate visualizations
    logger.info("   Generating error visualizations...")
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Convert signal lists back to arrays for visualization
    errors_for_viz = []
    for e in errors:
        e_copy = e.copy()
        e_copy['signal'] = np.array(e['signal'])
        errors_for_viz.append(e_copy)

    visualizations.visualize_misclassified_signals(
        errors=errors_for_viz,
        output_dir=viz_dir,
        class_names=['Normal', 'Arrhythmia'],
        max_examples_per_category=5,
        max_grid_examples=20
    )

    results = {
        'total_errors': len(errors),
        'false_positives_count': false_positives,
        'false_negatives_count': false_negatives,
        'error_categories': error_categories,
        'errors': errors  # Full error list (can be large)
    }

    return results


def run_task_2_4(
    model: nn.Module,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    ensemble_size: int,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
    max_samples: int = 200  # Limit for speed
) -> Dict[str, Any]:
    """
    Task 2.4: Robustness Testing

    Test model under noise and signal quality degradation

    Args:
        model: Trained model
        test_signals: Test signals
        test_labels: Test labels
        ensemble_size: Ensemble size
        device: Device
        output_dir: Output directory
        logger: Logger
        max_samples: Maximum samples to test (for speed)

    Returns:
        Dictionary with robustness testing results
    """
    logger.info(f"Running Task 2.4: Robustness Testing (max {max_samples} samples)...")

    # Prediction function for robustness testing
    def predict_fn(model, signal, device, ensemble_size):
        return ensemble_predict(
            model=model,
            input_data=signal,
            ensemble_size=ensemble_size,
            device=str(device),
            return_confidence=True
        )

    # Test additive noise robustness
    logger.info("   Testing additive noise robustness...")
    noise_results = robustness.test_additive_noise_robustness(
        model=model,
        test_signals=test_signals,
        test_labels=test_labels,
        predict_fn=predict_fn,
        snr_levels_db=[30, 20, 10],
        ensemble_size=ensemble_size,
        device=str(device),
        max_samples=max_samples
    )

    # Test signal quality variations
    logger.info("   Testing signal quality variations...")
    quality_results = robustness.test_signal_quality_variations(
        model=model,
        test_signals=test_signals,
        test_labels=test_labels,
        predict_fn=predict_fn,
        ensemble_size=ensemble_size,
        device=str(device),
        max_samples=max_samples
    )

    # Generate visualizations
    logger.info("   Generating robustness visualizations...")
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    visualizations.plot_noise_robustness(
        noise_results=noise_results,
        output_path=viz_dir / 'noise_robustness.png'
    )

    visualizations.plot_signal_quality_comparison(
        quality_results=quality_results,
        output_path=viz_dir / 'signal_quality_comparison.png'
    )

    results = {
        'noise_robustness': noise_results,
        'signal_quality': quality_results,
        'max_samples_tested': max_samples
    }

    # Log summary
    logger.info(f"   Clean accuracy: {noise_results.get('clean', {}).get('accuracy', 0):.1%}")
    for key in ['30dB', '20dB', '10dB']:
        if key in noise_results:
            acc = noise_results[key]['accuracy']
            logger.info(f"   {key} accuracy: {acc:.1%}")

    return results


def run_task_2_5(
    model: nn.Module,
    test_signals: np.ndarray,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Task 2.5: Performance Benchmarking

    Measure inference latency, throughput, and memory usage

    Args:
        model: Trained model
        test_signals: Test signals
        device: Device
        output_dir: Output directory
        logger: Logger

    Returns:
        Dictionary with performance benchmarks
    """
    logger.info("Running Task 2.5: Performance Benchmarking...")

    # Task 2.5.1: Latency Distribution
    logger.info("   Measuring latency distribution...")
    latency_stats = benchmark_inference_latency(
        model=model,
        test_signals=test_signals,
        device=device,
        n_trials=100,
        ensemble_sizes=[1, 3, 5]
    )

    # Task 2.5.2: Throughput Testing
    logger.info("   Measuring throughput...")
    throughput_data = benchmark_throughput(
        model=model,
        test_signals=test_signals,
        device=device,
        batch_sizes=[1, 4, 16, 32]
    )

    # Task 2.5.3: Memory Profiling
    logger.info("   Profiling memory usage...")
    memory_stats = profile_memory_usage(
        model=model,
        test_signal=test_signals[0],
        device=device
    )

    # Task 2.5.4: SNN Energy Metrics
    logger.info("   Measuring SNN energy metrics...")
    energy_stats = measure_snn_energy_metrics(
        model=model,
        test_signals=test_signals[:100],
        device=device
    )

    # Generate visualizations
    logger.info("   Generating performance visualizations...")
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    visualizations.plot_latency_distribution(
        latency_stats=latency_stats,
        output_path=viz_dir / 'latency_distribution.png'
    )

    visualizations.plot_throughput_comparison(
        throughput_data=throughput_data,
        output_path=viz_dir / 'throughput_comparison.png'
    )

    # Find optimal batch size
    optimal_bs = max(throughput_data, key=throughput_data.get)

    results = {
        'latency_distribution': latency_stats,
        'throughput': throughput_data,
        'optimal_batch_size': optimal_bs,
        'memory': memory_stats,
        'energy_metrics': energy_stats
    }

    # Log summary
    if 'single_inference' in latency_stats:
        logger.info(f"   Single inference mean: {latency_stats['single_inference']['mean']:.1f}ms")
    if 'ensemble_3' in latency_stats:
        logger.info(f"   Ensemble (N=3) mean: {latency_stats['ensemble_3']['mean']:.1f}ms")
    logger.info(f"   Optimal batch size: {optimal_bs}")
    logger.info(f"   Model size: {memory_stats['model_size_mb']:.2f} MB")

    return results


def benchmark_inference_latency(
    model: nn.Module,
    test_signals: np.ndarray,
    device: torch.device,
    n_trials: int = 100,
    ensemble_sizes: List[int] = [1, 3, 5]
) -> Dict[str, Dict[str, float]]:
    """Measure inference latency distribution"""
    results = {}

    for ensemble_size in ensemble_sizes:
        latencies = []

        # Warmup
        for _ in range(10):
            _ = ensemble_predict(
                model, test_signals[0], ensemble_size=ensemble_size, device=str(device)
            )

        # Measure
        for i in range(n_trials):
            signal = test_signals[i % len(test_signals)]

            start = time.perf_counter()
            _ = ensemble_predict(
                model, signal, ensemble_size=ensemble_size, device=str(device)
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        # Calculate percentiles
        key = 'single_inference' if ensemble_size == 1 else f'ensemble_{ensemble_size}'
        results[key] = {
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'mean': float(np.mean(latencies)),
            'median': float(np.median(latencies)),
            'std': float(np.std(latencies)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99))
        }

    return results


def benchmark_throughput(
    model: nn.Module,
    test_signals: np.ndarray,
    device: torch.device,
    batch_sizes: List[int] = [1, 4, 16, 32]
) -> Dict[int, float]:
    """Measure throughput at different batch sizes"""
    throughput = {}

    for batch_size in batch_sizes:
        # Warmup
        for _ in range(5):
            batch = test_signals[:batch_size]
            for signal in batch:
                _ = ensemble_predict(model, signal, ensemble_size=1, device=str(device))

        # Measure
        start = time.perf_counter()
        n_processed = 0

        for i in range(0, min(200, len(test_signals)), batch_size):
            batch = test_signals[i:i+batch_size]
            for signal in batch:
                _ = ensemble_predict(model, signal, ensemble_size=1, device=str(device))
                n_processed += 1

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        samples_per_sec = n_processed / elapsed
        throughput[batch_size] = float(samples_per_sec)

    return throughput


def profile_memory_usage(
    model: nn.Module,
    test_signal: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """Profile GPU/CPU memory usage"""
    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    # GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        _ = ensemble_predict(model, test_signal, ensemble_size=1, device=str(device))
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1e6
        _ = ensemble_predict(model, test_signal, ensemble_size=1, device=str(device))
        mem_after = process.memory_info().rss / 1e6
        peak_memory_mb = mem_after - mem_before

    return {
        'model_size_mb': float(model_size_mb),
        'peak_gpu_memory_mb': float(peak_memory_mb),
        'memory_per_sample_mb': float(peak_memory_mb)
    }


def measure_snn_energy_metrics(
    model: nn.Module,
    test_signals: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """Measure SNN-specific metrics: spike counts, sparsity"""
    spike_counts = []

    for signal in test_signals[:100]:
        result = ensemble_predict(
            model, signal, ensemble_size=1, device=str(device), return_confidence=True
        )
        spike_count = result.get('spike_count', 0)
        spike_counts.append(spike_count)

    mean_spikes = np.mean(spike_counts)

    # Estimate sparsity (percentage of active neurons)
    # SimpleSNN has 128 hidden neurons, 100 time steps
    total_possible_spikes = 128 * 100
    sparsity = mean_spikes / total_possible_spikes if total_possible_spikes > 0 else 0

    # Theoretical energy savings (from docs)
    theoretical_savings = 0.60  # 60% vs CNN

    return {
        'mean_spikes': float(mean_spikes),
        'sparsity': float(sparsity),
        'theoretical_energy_savings': float(theoretical_savings)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Comprehensive Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tasks
    python scripts/comprehensive_evaluation.py

    # Skip robustness testing (faster)
    python scripts/comprehensive_evaluation.py --skip-tasks 2.4

    # Use smaller ensemble for speed
    python scripts/comprehensive_evaluation.py --ensemble-size 1

    # Test on subset
    python scripts/comprehensive_evaluation.py --max-samples 500
        """
    )
    parser.add_argument('--model-path', default='models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--test-data', default='data/synthetic/test_data.pt',
                       help='Path to test dataset')
    parser.add_argument('--ensemble-size', type=int, default=3,
                       help='Ensemble size for predictions (default: 3)')
    parser.add_argument('--output-dir', default='results/phase2_evaluation',
                       help='Output directory for results')
    parser.add_argument('--skip-tasks', nargs='*', choices=['2.1', '2.2', '2.3', '2.4', '2.5'],
                       help='Tasks to skip')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum test samples to evaluate (None = all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_file=output_dir / 'logs' / 'evaluation.log')

    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info("PHASE 2: COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test Data: {args.test_data}")
    logger.info(f"Ensemble Size: {args.ensemble_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Random Seed: {args.seed}")

    # Load model and data
    logger.info("\n📦 Loading model and test data...")
    try:
        # Detect and instantiate correct model architecture
        model_instance = detect_model_architecture(args.model_path, device=str(device))
        model = load_model(args.model_path, model_instance, device=str(device))
        logger.info(f"   ✅ Model loaded successfully ({model_instance.__class__.__name__})")
    except Exception as e:
        logger.error(f"   ❌ Failed to load model: {e}")
        return 1

    try:
        test_data = torch.load(args.test_data)
        test_signals = test_data['signals'].numpy()
        test_labels = test_data['labels'].numpy()

        if args.max_samples is not None:
            test_signals = test_signals[:args.max_samples]
            test_labels = test_labels[:args.max_samples]

        logger.info(f"   ✅ Loaded {len(test_signals)} test samples")
    except Exception as e:
        logger.error(f"   ❌ Failed to load test data: {e}")
        return 1

    results = {}
    skip_tasks = args.skip_tasks or []

    # Task 2.1: Full Test Set Analysis
    if '2.1' not in skip_tasks:
        logger.info("\n" + "=" * 70)
        logger.info("TASK 2.1: Full Test Set Analysis")
        logger.info("=" * 70)
        try:
            results['task_2_1'] = run_task_2_1(
                model, test_signals, test_labels, args.ensemble_size, device, logger
            )
            save_results(results['task_2_1'], output_dir / 'metrics' / 'task_2_1_test_set_analysis.json')
        except Exception as e:
            logger.error(f"Task 2.1 failed: {e}", exc_info=True)
            return 1

    # Task 2.2: Clinical Metrics
    if '2.2' not in skip_tasks and 'task_2_1' in results:
        logger.info("\n" + "=" * 70)
        logger.info("TASK 2.2: Clinical Performance Metrics")
        logger.info("=" * 70)
        try:
            results['task_2_2'] = run_task_2_2(results['task_2_1'], logger)
            save_results(results['task_2_2'], output_dir / 'metrics' / 'task_2_2_clinical_metrics.json')

            # Generate confusion matrix visualization
            cm = np.array(results['task_2_2']['confusion_matrix'])
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            visualizations.plot_confusion_matrix(
                cm, ['Normal', 'Arrhythmia'],
                viz_dir / 'confusion_matrix.png',
                title="Test Set Confusion Matrix"
            )
        except Exception as e:
            logger.error(f"Task 2.2 failed: {e}", exc_info=True)
            return 1

    # Task 2.3: Error Analysis
    if '2.3' not in skip_tasks and 'task_2_1' in results:
        logger.info("\n" + "=" * 70)
        logger.info("TASK 2.3: Error Pattern Analysis")
        logger.info("=" * 70)
        try:
            results['task_2_3'] = run_task_2_3(
                results['task_2_1'], test_signals, output_dir, logger
            )
            save_results(results['task_2_3'], output_dir / 'metrics' / 'task_2_3_error_analysis.json')
        except Exception as e:
            logger.error(f"Task 2.3 failed: {e}", exc_info=True)
            return 1

    # Task 2.4: Robustness Testing
    if '2.4' not in skip_tasks:
        logger.info("\n" + "=" * 70)
        logger.info("TASK 2.4: Robustness Testing")
        logger.info("=" * 70)
        try:
            results['task_2_4'] = run_task_2_4(
                model, test_signals, test_labels, args.ensemble_size,
                device, output_dir, logger, max_samples=200
            )
            save_results(results['task_2_4'], output_dir / 'metrics' / 'task_2_4_robustness.json')
        except Exception as e:
            logger.error(f"Task 2.4 failed: {e}", exc_info=True)
            # Continue even if robustness testing fails

    # Task 2.5: Performance Benchmarking
    if '2.5' not in skip_tasks:
        logger.info("\n" + "=" * 70)
        logger.info("TASK 2.5: Performance Benchmarking")
        logger.info("=" * 70)
        try:
            results['task_2_5'] = run_task_2_5(
                model, test_signals, device, output_dir, logger
            )
            save_results(results['task_2_5'], output_dir / 'metrics' / 'task_2_5_performance.json')
        except Exception as e:
            logger.error(f"Task 2.5 failed: {e}", exc_info=True)
            # Continue even if benchmarking fails

    # Generate comprehensive report
    logger.info("\n" + "=" * 70)
    logger.info("📝 GENERATING COMPREHENSIVE REPORT")
    logger.info("=" * 70)
    try:
        report_path = project_root / 'docs' / 'PHASE2_EVALUATION_REPORT.md'
        reports.generate_comprehensive_report(results, report_path)
        logger.info(f"   ✅ Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Report saved to: {project_root / 'docs' / 'PHASE2_EVALUATION_REPORT.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
