"""
Clinical Metrics Module
========================

Comprehensive clinical performance metrics for medical AI validation.

Owner: Phase 2 Implementation
Date: 2025-11-09
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def calculate_comprehensive_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate all clinical validation metrics for binary classification

    Args:
        y_true: True labels (0=Normal, 1=Arrhythmia)
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for AUC)
        class_names: Class names for reporting

    Returns:
        Dictionary containing:
            - binary_metrics: Sensitivity, Specificity, PPV, NPV, F1, Accuracy
            - confusion_matrix: 2x2 matrix
            - targets: Target thresholds and status
            - actionable_insights: List of recommendations
            - clinical_interpretation: Summary text

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 0, 1])
        >>> metrics = calculate_comprehensive_clinical_metrics(y_true, y_pred)
        >>> print(f"Sensitivity: {metrics['binary_metrics']['sensitivity']:.1%}")
    """
    if class_names is None:
        class_names = ['Normal', 'Arrhythmia']

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract components
    # For binary: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    # Calculate binary metrics (handle division by zero)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Recall, True Positive Rate
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # True Negative Rate
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0    # Positive Predictive Value
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0          # Negative Predictive Value
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # F1 score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # False rates
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    binary_metrics = {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'ppv': float(precision),  # Alias
        'npv': float(npv),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate)
    }

    # Calculate AUC if probabilities provided
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            binary_metrics['auc_roc'] = float(auc)
        except:
            binary_metrics['auc_roc'] = None

    # Define clinical targets (from docs/NEXT_STEPS_DETAILED.md lines 203-225)
    targets = {
        'sensitivity_target': 0.95,
        'sensitivity_met': sensitivity >= 0.95,
        'specificity_target': 0.90,
        'specificity_met': specificity >= 0.90,
        'precision_target': 0.85,
        'precision_met': precision >= 0.85,
        'npv_target': 0.95,
        'npv_met': npv >= 0.95
    }

    # Generate actionable insights
    actionable_insights = generate_actionable_insights(
        binary_metrics, targets, cm, class_names
    )

    # Clinical interpretation
    interpretation = generate_clinical_interpretation(
        binary_metrics, targets, TN, FP, FN, TP
    )

    return {
        'binary_metrics': binary_metrics,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_raw': {'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)},
        'targets': targets,
        'actionable_insights': actionable_insights,
        'clinical_interpretation': interpretation,
        'class_names': class_names
    }


def generate_actionable_insights(
    metrics: Dict[str, float],
    targets: Dict[str, Any],
    confusion_matrix: np.ndarray,
    class_names: List[str]
) -> List[str]:
    """
    Generate specific recommendations for model improvement

    Args:
        metrics: Binary metrics dictionary
        targets: Target thresholds and status
        confusion_matrix: 2x2 confusion matrix
        class_names: Class names

    Returns:
        List of actionable insight strings
    """
    insights = []

    TN, FP, FN, TP = confusion_matrix.ravel()
    total = TN + FP + FN + TP

    # Sensitivity analysis
    if not targets['sensitivity_met']:
        sensitivity = metrics['sensitivity']
        miss_rate = metrics['false_negative_rate']
        insights.append(
            f"⚠️  **CRITICAL**: Sensitivity {sensitivity:.1%} < {targets['sensitivity_target']:.0%} target"
        )
        insights.append(
            f"    → Missing {miss_rate:.1%} of {class_names[1]} cases ({FN}/{FN+TP} false negatives)"
        )
        insights.append(
            f"    → **Recommendation**: Lower classification threshold (accept more false alarms)"
        )
        insights.append(
            f"    → **Recommendation**: Increase {class_names[1]} training samples by 2-3x"
        )
        insights.append(
            f"    → **Recommendation**: Add data augmentation for {class_names[1]} class"
        )
        insights.append(
            f"    → **Recommendation**: Consider class-weighted loss function"
        )
    else:
        insights.append(
            f"✅ Sensitivity {metrics['sensitivity']:.1%} meets {targets['sensitivity_target']:.0%} target"
        )

    # Specificity analysis
    if not targets['specificity_met']:
        specificity = metrics['specificity']
        false_alarm_rate = metrics['false_positive_rate']
        insights.append(
            f"⚠️  Specificity {specificity:.1%} < {targets['specificity_target']:.0%} target"
        )
        insights.append(
            f"    → {false_alarm_rate:.1%} false alarm rate ({FP}/{FP+TN} false positives)"
        )
        insights.append(
            f"    → **Recommendation**: Increase classification threshold"
        )
        insights.append(
            f"    → **Recommendation**: Improve feature discrimination between classes"
        )
    else:
        insights.append(
            f"✅ Specificity {metrics['specificity']:.1%} meets {targets['specificity_target']:.0%} target"
        )

    # Precision/PPV analysis
    if not targets['precision_met']:
        precision = metrics['precision']
        insights.append(
            f"⚠️  Precision (PPV) {precision:.1%} < {targets['precision_target']:.0%} target"
        )
        insights.append(
            f"    → When predicting {class_names[1]}, only {precision:.1%} are correct"
        )
        insights.append(
            f"    → **Recommendation**: Reduce false positives via threshold tuning"
        )
    else:
        insights.append(
            f"✅ Precision (PPV) {metrics['precision']:.1%} meets {targets['precision_target']:.0%} target"
        )

    # NPV analysis
    if not targets['npv_met']:
        npv = metrics['npv']
        insights.append(
            f"⚠️  **CRITICAL**: NPV {npv:.1%} < {targets['npv_target']:.0%} target"
        )
        insights.append(
            f"    → When predicting {class_names[0]}, {1-npv:.1%} are actually {class_names[1]}"
        )
        insights.append(
            f"    → **Clinical Risk**: Patients sent home may have undetected {class_names[1]}"
        )
        insights.append(
            f"    → **Recommendation**: Improve sensitivity (reduce false negatives)"
        )
    else:
        insights.append(
            f"✅ NPV {metrics['npv']:.1%} meets {targets['npv_target']:.0%} target"
        )

    # Overall assessment
    insights.append("")
    insights.append("**Overall Assessment**:")
    targets_met = sum([
        targets['sensitivity_met'],
        targets['specificity_met'],
        targets['precision_met'],
        targets['npv_met']
    ])

    if targets_met == 4:
        insights.append("🎉 All clinical targets met - Model ready for deployment consideration")
    elif targets_met >= 2:
        insights.append(f"⚠️  {targets_met}/4 clinical targets met - Needs improvement before deployment")
    else:
        insights.append(f"🔴 Only {targets_met}/4 clinical targets met - NOT READY for clinical use")

    return insights


def generate_clinical_interpretation(
    metrics: Dict[str, float],
    targets: Dict[str, Any],
    TN: int, FP: int, FN: int, TP: int
) -> str:
    """
    Generate human-readable clinical interpretation

    Args:
        metrics: Binary metrics dictionary
        targets: Targets and status
        TN, FP, FN, TP: Confusion matrix components

    Returns:
        Multi-line interpretation string
    """
    total = TN + FP + FN + TP

    interpretation = f"""
## Clinical Performance Interpretation

**Dataset Size**: {total} samples
**Correct Predictions**: {TN + TP} ({(TN+TP)/total:.1%})
**Incorrect Predictions**: {FP + FN} ({(FP+FN)/total:.1%})

**Confusion Matrix**:
```
                    Predicted
                Normal    Arrhythmia
True Normal       {TN:4d}      {FP:4d}
     Arrhythmia   {FN:4d}      {TP:4d}
```

**Clinical Metrics**:
- **Sensitivity** (Recall): {metrics['sensitivity']:.1%}
  - Ability to detect arrhythmia cases
  - Target: ≥95% (CRITICAL for patient safety)
  - Status: {'✅ PASS' if targets['sensitivity_met'] else '❌ FAIL'}

- **Specificity**: {metrics['specificity']:.1%}
  - Ability to correctly identify normal cases
  - Target: ≥90% (reduces alarm fatigue)
  - Status: {'✅ PASS' if targets['specificity_met'] else '❌ FAIL'}

- **Positive Predictive Value** (Precision): {metrics['precision']:.1%}
  - When model says "arrhythmia", how often is it correct?
  - Target: ≥85% (clinical trust)
  - Status: {'✅ PASS' if targets['precision_met'] else '❌ FAIL'}

- **Negative Predictive Value**: {metrics['npv']:.1%}
  - When model says "normal", how often is it correct?
  - Target: ≥95% (safety - patients sent home must be normal)
  - Status: {'✅ PASS' if targets['npv_met'] else '❌ FAIL'}

**Key Performance Indicators**:
- False Negative Rate: {metrics['false_negative_rate']:.1%} ({FN} missed arrhythmias)
- False Positive Rate: {metrics['false_positive_rate']:.1%} ({FP} false alarms)
- F1-Score: {metrics['f1_score']:.3f}
- Overall Accuracy: {metrics['accuracy']:.1%}

**Clinical Deployment Readiness**:
{_get_deployment_status(targets)}
"""

    return interpretation


def _get_deployment_status(targets: Dict[str, Any]) -> str:
    """Helper to generate deployment readiness assessment"""
    targets_met = sum([
        targets['sensitivity_met'],
        targets['specificity_met'],
        targets['precision_met'],
        targets['npv_met']
    ])

    if targets_met == 4:
        return "✅ **READY**: All 4 clinical targets met. Model suitable for controlled clinical trials."
    elif targets_met == 3:
        return "⚠️  **NEEDS MINOR IMPROVEMENTS**: 3/4 targets met. Address failing metric before deployment."
    elif targets_met == 2:
        return "⚠️  **NEEDS SIGNIFICANT IMPROVEMENTS**: 2/4 targets met. Not ready for clinical use."
    else:
        return "🔴 **NOT READY**: Fewer than half of clinical targets met. Requires major model improvements."


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy, precision, recall, F1 for each class

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names

    Returns:
        Dictionary with per-class metrics
    """
    if class_names is None:
        class_names = ['Normal', 'Arrhythmia']

    # Use sklearn classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics = {}
    for class_name in class_names:
        if class_name in report:
            per_class_metrics[class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': int(report[class_name]['support'])
            }

    # Add overall metrics
    per_class_metrics['macro_avg'] = {
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score']
    }

    per_class_metrics['weighted_avg'] = {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }

    return per_class_metrics
