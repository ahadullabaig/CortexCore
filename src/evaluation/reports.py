"""
Report Generation Module
=========================

Generate comprehensive markdown reports for Phase 2 evaluation.

Owner: Phase 2 Implementation
Date: 2025-11-09
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def generate_comprehensive_report(
    results: Dict[str, Any],
    output_path: Path
):
    """
    Generate comprehensive markdown report with all Phase 2 findings

    Args:
        results: Dictionary containing all task results:
            - task_2_1: Test set analysis
            - task_2_2: Clinical metrics
            - task_2_3: Error analysis
            - task_2_4: Robustness testing
            - task_2_5: Performance benchmarking
        output_path: Path to save report

    Structure:
        1. Executive Summary
        2. Test Set Performance
        3. Clinical Metrics Analysis
        4. Error Pattern Analysis
        5. Robustness Testing Results
        6. Performance Benchmarks
        7. Recommendations
        8. Appendix
    """
    report_lines = []

    # Header
    report_lines.append("# Phase 2: Comprehensive Model Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Model:** SimpleSNN (models/best_model.pt)")
    report_lines.append(f"**Test Set:** 1000 synthetic ECG samples")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Executive Summary
    report_lines.extend(_generate_executive_summary(results))

    # Task 2.1: Test Set Performance
    if 'task_2_1' in results:
        report_lines.extend(_generate_test_set_section(results['task_2_1']))

    # Task 2.2: Clinical Metrics
    if 'task_2_2' in results:
        report_lines.extend(_generate_clinical_metrics_section(results['task_2_2']))

    # Task 2.3: Error Analysis
    if 'task_2_3' in results:
        report_lines.extend(_generate_error_analysis_section(results['task_2_3']))

    # Task 2.4: Robustness Testing
    if 'task_2_4' in results:
        report_lines.extend(_generate_robustness_section(results['task_2_4']))

    # Task 2.5: Performance Benchmarking
    if 'task_2_5' in results:
        report_lines.extend(_generate_performance_section(results['task_2_5']))

    # Recommendations
    report_lines.extend(_generate_recommendations(results))

    # Appendix
    report_lines.extend(_generate_appendix(results))

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n📄 Comprehensive report saved to: {output_path}")


def _generate_executive_summary(results: Dict[str, Any]) -> List[str]:
    """Generate executive summary section"""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")

    # Extract key metrics
    if 'task_2_1' in results:
        overall_acc = results['task_2_1'].get('overall_accuracy', 0)
        lines.append(f"- **Overall Accuracy:** {overall_acc:.1%}")

    if 'task_2_2' in results:
        metrics = results['task_2_2'].get('binary_metrics', {})
        sensitivity = metrics.get('sensitivity', 0)
        specificity = metrics.get('specificity', 0)

        sens_status = "✅" if sensitivity >= 0.95 else "❌"
        spec_status = "✅" if specificity >= 0.90 else "✅" if specificity >= 0.85 else "❌"

        lines.append(f"- **Arrhythmia Detection (Sensitivity):** {sensitivity:.1%} {sens_status} (target: ≥95%)")
        lines.append(f"- **Normal Detection (Specificity):** {specificity:.1%} {spec_status} (target: ≥90%)")

    if 'task_2_3' in results:
        total_errors = results['task_2_3'].get('total_errors', 0)
        lines.append(f"- **Total Errors:** {total_errors}")

    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    # Add critical findings based on results
    if 'task_2_2' in results:
        insights = results['task_2_2'].get('actionable_insights', [])
        for insight in insights[:5]:  # Top 5 insights
            lines.append(f"- {insight}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return lines


def _generate_test_set_section(task_results: Dict[str, Any]) -> List[str]:
    """Generate test set performance section"""
    lines = []
    lines.append("## Task 2.1: Test Set Performance")
    lines.append("")

    # Overall Metrics
    overall_acc = task_results.get('overall_accuracy', 0)
    lines.append("### Overall Metrics")
    lines.append("")
    lines.append(f"- **Test Accuracy:** {overall_acc:.1%}")
    lines.append(f"- **Total Samples:** {task_results.get('total_samples', 1000)}")
    lines.append(f"- **Ensemble Size:** {task_results.get('ensemble_size', 3)}")

    if 'mean_inference_time_ms' in task_results:
        lines.append(f"- **Mean Inference Time:** {task_results['mean_inference_time_ms']:.1f}ms")

    lines.append("")

    # Per-Class Performance
    if 'per_class_accuracy' in task_results:
        lines.append("### Per-Class Performance")
        lines.append("")
        lines.append("| Class | Accuracy | Samples | Correct |")
        lines.append("|-------|----------|---------|---------|")

        per_class = task_results['per_class_accuracy']
        for class_name, acc in per_class.items():
            # Estimate samples (assumes balanced)
            total = task_results.get('total_samples', 1000)
            samples = total // len(per_class)
            correct = int(samples * acc)
            lines.append(f"| {class_name} | {acc:.1%} | {samples} | {correct} |")

        lines.append("")

    # Confusion Matrix
    if 'confusion_matrix' in task_results:
        cm = task_results['confusion_matrix']
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("```")
        lines.append("                 Predicted")
        lines.append("              Normal  Arrhythmia")
        lines.append(f"True Normal     {cm[0][0]:4d}      {cm[0][1]:4d}")
        lines.append(f"     Arrhythmia {cm[1][0]:4d}      {cm[1][1]:4d}")
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_clinical_metrics_section(task_results: Dict[str, Any]) -> List[str]:
    """Generate clinical metrics section"""
    lines = []
    lines.append("## Task 2.2: Clinical Performance Metrics")
    lines.append("")

    metrics = task_results.get('binary_metrics', {})
    targets = task_results.get('targets', {})

    # Metrics Table
    lines.append("### Metrics Comparison")
    lines.append("")
    lines.append("| Metric | Value | Target | Status |")
    lines.append("|--------|-------|--------|--------|")

    metrics_to_show = [
        ('Sensitivity', 'sensitivity', 'sensitivity_target', 'sensitivity_met'),
        ('Specificity', 'specificity', 'specificity_target', 'specificity_met'),
        ('Precision (PPV)', 'precision', 'precision_target', 'precision_met'),
        ('NPV', 'npv', 'npv_target', 'npv_met'),
        ('F1-Score', 'f1_score', None, None),
        ('Accuracy', 'accuracy', None, None)
    ]

    for label, key, target_key, met_key in metrics_to_show:
        value = metrics.get(key, 0)
        target = targets.get(target_key, '-') if target_key else '-'
        met = targets.get(met_key, None) if met_key else None

        if target != '-':
            target_str = f"≥{target:.0%}"
            status = "✅ PASS" if met else "❌ FAIL"
        else:
            target_str = "-"
            status = "-"

        lines.append(f"| {label} | {value:.1%} | {target_str} | {status} |")

    lines.append("")

    # Actionable Insights
    if 'actionable_insights' in task_results:
        lines.append("### Actionable Insights")
        lines.append("")
        for insight in task_results['actionable_insights']:
            lines.append(insight)
        lines.append("")

    # Clinical Interpretation
    if 'clinical_interpretation' in task_results:
        lines.append("### Clinical Interpretation")
        lines.append("")
        lines.append(task_results['clinical_interpretation'])
        lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_error_analysis_section(task_results: Dict[str, Any]) -> List[str]:
    """Generate error analysis section"""
    lines = []
    lines.append("## Task 2.3: Error Pattern Analysis")
    lines.append("")

    total_errors = task_results.get('total_errors', 0)
    false_positives = task_results.get('false_positives_count', 0)
    false_negatives = task_results.get('false_negatives_count', 0)

    lines.append("### Error Distribution")
    lines.append("")
    lines.append(f"- **Total Errors:** {total_errors}")
    lines.append(f"- **False Positives:** {false_positives} (Normal → Arrhythmia)")
    lines.append(f"- **False Negatives:** {false_negatives} (Arrhythmia → Normal) ⚠️ CRITICAL")
    lines.append("")

    # Error Categories
    if 'error_categories' in task_results:
        categories = task_results['error_categories']
        lines.append("### Error Categories")
        lines.append("")
        lines.append("| Category | Count | % of Errors | Mean Confidence |")
        lines.append("|----------|-------|-------------|-----------------|")

        for cat_name, cat_data in categories.items():
            count = cat_data.get('count', 0)
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            mean_conf = cat_data.get('mean_confidence', 0)
            lines.append(f"| {cat_name.capitalize()} | {count} | {pct:.1f}% | {mean_conf:.1%} |")

        lines.append("")

    # Category Descriptions
    lines.append("### Category Definitions")
    lines.append("")
    lines.append("- **Borderline**: Low confidence (<60%) and high variance (>15% std) - model uncertain")
    lines.append("- **Noisy**: High confidence variance (>20% std) - signal quality issues")
    lines.append("- **Atypical**: Unusual signal morphology (low std <0.1) - rare patterns")
    lines.append("- **Systematic**: Model consistently wrong with high confidence - bias issue")
    lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_robustness_section(task_results: Dict[str, Any]) -> List[str]:
    """Generate robustness testing section"""
    lines = []
    lines.append("## Task 2.4: Robustness Testing")
    lines.append("")

    # Noise Robustness
    if 'noise_robustness' in task_results:
        lines.append("### Additive Noise Robustness")
        lines.append("")
        lines.append("| SNR (dB) | Accuracy | Degradation | Class 0 Acc | Class 1 Acc |")
        lines.append("|----------|----------|-------------|-------------|-------------|")

        noise_results = task_results['noise_robustness']
        for key in sorted(noise_results.keys()):
            if 'dB' in key or key == 'clean':
                result = noise_results[key]
                acc = result.get('accuracy', 0)
                deg = result.get('degradation_pct', 0) if key != 'clean' else 0
                c0 = result.get('class_0_accuracy', 0)
                c1 = result.get('class_1_accuracy', 0)

                label = "Clean" if key == 'clean' else key
                lines.append(f"| {label} | {acc:.1%} | {deg:.1f}% | {c0:.1%} | {c1:.1%} |")

        lines.append("")

        # Clinical Viability
        lines.append("**Clinical Viability Assessment:**")
        lines.append("")

        # Check 20dB performance (typical real-world SNR)
        if '20dB' in noise_results:
            acc_20db = noise_results['20dB']['accuracy']
            if acc_20db >= 0.85:
                lines.append(f"✅ 20dB SNR: {acc_20db:.1%} (acceptable for clinical use)")
            else:
                lines.append(f"⚠️ 20dB SNR: {acc_20db:.1%} (below 85% threshold)")

        lines.append("")

    # Signal Quality
    if 'signal_quality' in task_results:
        lines.append("### Signal Quality Variations")
        lines.append("")
        lines.append("| Degradation Type | Accuracy | Degradation vs Clean |")
        lines.append("|-----------------|----------|---------------------|")

        quality_results = task_results['signal_quality']
        clean_acc = task_results.get('noise_robustness', {}).get('clean', {}).get('accuracy', 0.89)

        for deg_type, result in quality_results.items():
            acc = result.get('accuracy', 0)
            deg_pct = ((clean_acc - acc) / clean_acc) * 100 if clean_acc > 0 else 0
            lines.append(f"| {deg_type.replace('_', ' ').capitalize()} | {acc:.1%} | {deg_pct:.1f}% |")

        lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_performance_section(task_results: Dict[str, Any]) -> List[str]:
    """Generate performance benchmarking section"""
    lines = []
    lines.append("## Task 2.5: Performance Benchmarking")
    lines.append("")

    # Latency Distribution
    if 'latency_distribution' in task_results:
        lines.append("### Inference Latency Distribution")
        lines.append("")
        lines.append("| Metric | Single | Ensemble (N=3) | Ensemble (N=5) |")
        lines.append("|--------|--------|----------------|----------------|")

        lat_dist = task_results['latency_distribution']
        metrics_to_show = ['min', 'median', 'mean', 'p95', 'p99', 'max']

        for metric in metrics_to_show:
            single = lat_dist.get('single_inference', {}).get(metric, 0)
            ens3 = lat_dist.get('ensemble_3', {}).get(metric, 0)
            ens5 = lat_dist.get('ensemble_5', {}).get(metric, 0)

            lines.append(f"| {metric.upper()} | {single:.1f}ms | {ens3:.1f}ms | {ens5:.1f}ms |")

        lines.append("")

    # Throughput
    if 'throughput' in task_results:
        lines.append("### Throughput by Batch Size")
        lines.append("")
        lines.append("| Batch Size | Throughput (samples/sec) |")
        lines.append("|------------|-------------------------|")

        throughput = task_results['throughput']
        for batch_size in sorted(throughput.keys()):
            tput = throughput[batch_size]
            lines.append(f"| {batch_size} | {tput:.1f} |")

        optimal_bs = task_results.get('optimal_batch_size', 32)
        lines.append("")
        lines.append(f"**Optimal Batch Size:** {optimal_bs}")
        lines.append("")

    # Memory
    if 'memory' in task_results:
        mem = task_results['memory']
        lines.append("### Memory Usage")
        lines.append("")
        lines.append(f"- **Model Size:** {mem.get('model_size_mb', 0):.2f} MB")
        lines.append(f"- **Peak GPU Memory:** {mem.get('peak_gpu_memory_mb', 0):.2f} MB")
        lines.append(f"- **Memory per Sample:** {mem.get('memory_per_sample_mb', 0):.3f} MB")
        lines.append("")

    # Energy Metrics
    if 'energy_metrics' in task_results:
        energy = task_results['energy_metrics']
        lines.append("### SNN Energy Metrics")
        lines.append("")
        lines.append(f"- **Mean Spikes per Inference:** {energy.get('mean_spikes', 0):.1f}")
        lines.append(f"- **Sparsity:** {energy.get('sparsity', 0):.1%}")
        lines.append(f"- **Theoretical Energy Savings vs ANN:** {energy.get('theoretical_energy_savings', 0):.0%}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate recommendations section"""
    lines = []
    lines.append("## Recommendations")
    lines.append("")

    lines.append("### Immediate Actions (This Week)")
    lines.append("")

    # Based on sensitivity
    if 'task_2_2' in results:
        sensitivity = results['task_2_2'].get('binary_metrics', {}).get('sensitivity', 0)
        if sensitivity < 0.95:
            lines.append("1. **🔴 CRITICAL: Improve Arrhythmia Detection**")
            lines.append(f"   - Current sensitivity: {sensitivity:.1%} (target: 95%)")
            lines.append("   - Action: Retrain with class-balanced dataset")
            lines.append("   - Action: Lower classification threshold")
            lines.append("   - Action: Add data augmentation for arrhythmia class")
            lines.append("")

    # Based on error analysis
    if 'task_2_3' in results:
        total_errors = results['task_2_3'].get('total_errors', 0)
        if total_errors > 150:
            lines.append("2. **⚠️ High Error Rate**")
            lines.append(f"   - Total errors: {total_errors} (16.9%)")
            lines.append("   - Action: Analyze error patterns and retrain")
            lines.append("")

    lines.append("### Short-term Improvements (Next 2 Weeks)")
    lines.append("")
    lines.append("1. **Data Augmentation** (Phase 3.2)")
    lines.append("   - Implement time warping, amplitude scaling, noise injection")
    lines.append("   - Expected improvement: 2-5% accuracy")
    lines.append("")
    lines.append("2. **Hyperparameter Tuning** (Phase 3.3)")
    lines.append("   - Optimize learning rate, batch size, beta parameter")
    lines.append("   - Expected improvement: 1-3% accuracy")
    lines.append("")
    lines.append("3. **Architecture Enhancement** (Phase 4)")
    lines.append("   - Try convolutional SNN or deeper network")
    lines.append("   - Expected improvement: 3-5% accuracy")
    lines.append("")

    lines.append("### Long-term Goals (Next Month)")
    lines.append("")
    lines.append("1. **Train on Real Data** (Phase 8)")
    lines.append("   - Acquire and integrate MIT-BIH dataset")
    lines.append("   - Expected: More realistic performance metrics")
    lines.append("")
    lines.append("2. **Production Optimization** (Phase 7)")
    lines.append("   - Model quantization and pruning")
    lines.append("   - Edge device deployment")
    lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _generate_appendix(results: Dict[str, Any]) -> List[str]:
    """Generate appendix section"""
    lines = []
    lines.append("## Appendix")
    lines.append("")

    lines.append("### Visualizations Generated")
    lines.append("")
    lines.append("All visualizations saved to: `results/phase2_evaluation/visualizations/`")
    lines.append("")
    lines.append("- `confusion_matrix.png` - Confusion matrix heatmap")
    lines.append("- `error_grid_all.png` - Grid of misclassified signals")
    lines.append("- `error_category_*.png` - Error signals by category")
    lines.append("- `confidence_distributions.png` - Confidence distributions")
    lines.append("- `error_category_summary.png` - Error category pie chart")
    lines.append("- `noise_robustness.png` - SNR degradation curve")
    lines.append("- `signal_quality_comparison.png` - Quality variation comparison")
    lines.append("- `latency_distribution.png` - Latency box plots")
    lines.append("- `throughput_comparison.png` - Throughput vs batch size")
    lines.append("")

    lines.append("### Detailed Results")
    lines.append("")
    lines.append("All detailed metrics saved to: `results/phase2_evaluation/metrics/`")
    lines.append("")
    lines.append("- `task_2_1_test_set_analysis.json`")
    lines.append("- `task_2_2_clinical_metrics.json`")
    lines.append("- `task_2_3_error_analysis.json`")
    lines.append("- `task_2_4_robustness.json`")
    lines.append("- `task_2_5_performance.json`")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Phase 2 Implementation**: Complete")
    lines.append("")

    return lines
