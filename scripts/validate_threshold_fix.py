"""
Quick Validation Script for Threshold Implementation (Fix #1)
============================================================

Tests that the calibrated threshold is working correctly by:
1. Loading the model and test data
2. Running predictions with and without threshold
3. Comparing results to verify expected behavior

Usage:
    python scripts/validate_threshold_fix.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SimpleSNN
from src.inference import load_model, predict
from sklearn.metrics import confusion_matrix


def test_threshold_implementation():
    """Test that threshold implementation works as expected"""
    print("=" * 80)
    print("THRESHOLD IMPLEMENTATION VALIDATION")
    print("=" * 80)

    # Load model
    print("\n📦 Loading model...")
    device = 'cpu'
    model = SimpleSNN()
    model = load_model('models/best_model.pt', model, device=device)
    print("✅ Model loaded")

    # Load test data (use small subset for quick test)
    print("\n📦 Loading test data...")
    test_data = torch.load('data/synthetic/test_data.pt')
    signals = test_data['signals'][:100]  # Just 100 samples for quick test
    labels = test_data['labels'][:100]
    print(f"✅ Loaded {len(signals)} test samples")

    # Test 1: Baseline (no threshold = argmax)
    print("\n" + "=" * 80)
    print("TEST 1: Baseline (no threshold parameter)")
    print("=" * 80)
    y_true = []
    y_pred_baseline = []

    for i in range(len(signals)):
        result = predict(model, signals[i].numpy(), device=device, seed=42)
        y_true.append(labels[i].item())
        y_pred_baseline.append(result['prediction'])

    cm_baseline = confusion_matrix(y_true, y_pred_baseline)
    tn, fp, fn, tp = cm_baseline.ravel()
    sensitivity_baseline = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_baseline = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Arrhythmia")
    print(f"True Normal      {tn:4d}      {fp:4d}")
    print(f"     Arrhythmia  {fn:4d}      {tp:4d}")
    print(f"\nSensitivity: {sensitivity_baseline:.1%}")
    print(f"Specificity: {specificity_baseline:.1%}")

    # Test 2: With threshold = 0.40 (target high sensitivity)
    print("\n" + "=" * 80)
    print("TEST 2: With sensitivity_threshold=0.40")
    print("=" * 80)
    y_pred_threshold = []
    arrhythmia_probs = []

    for i in range(len(signals)):
        result = predict(model, signals[i].numpy(), device=device, seed=42,
                        sensitivity_threshold=0.40)
        y_pred_threshold.append(result['prediction'])
        arrhythmia_probs.append(result['arrhythmia_probability'])

    cm_threshold = confusion_matrix(y_true, y_pred_threshold)
    tn2, fp2, fn2, tp2 = cm_threshold.ravel()
    sensitivity_threshold = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    specificity_threshold = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Arrhythmia")
    print(f"True Normal      {tn2:4d}      {fp2:4d}")
    print(f"     Arrhythmia  {fn2:4d}      {tp2:4d}")
    print(f"\nSensitivity: {sensitivity_threshold:.1%}")
    print(f"Specificity: {specificity_threshold:.1%}")

    # Verify arrhythmia_probability is returned
    print(f"\n✅ arrhythmia_probability returned: {len(arrhythmia_probs)} values")
    print(f"   Range: [{min(arrhythmia_probs):.3f}, {max(arrhythmia_probs):.3f}]")

    # Test 3: Verify threshold logic
    print("\n" + "=" * 80)
    print("TEST 3: Verify threshold logic")
    print("=" * 80)

    # Test with threshold=0.3 (very sensitive)
    result_low = predict(model, signals[0].numpy(), device=device, seed=42,
                         sensitivity_threshold=0.3)

    # Test with threshold=0.6 (very specific)
    result_high = predict(model, signals[0].numpy(), device=device, seed=42,
                          sensitivity_threshold=0.6)

    print(f"\nSame signal, different thresholds:")
    print(f"  Arrhythmia probability: {result_low['arrhythmia_probability']:.3f}")
    print(f"  Threshold 0.3 → Prediction: {result_low['prediction']} ({result_low['class_name']})")
    print(f"  Threshold 0.6 → Prediction: {result_high['prediction']} ({result_high['class_name']})")

    if result_low['arrhythmia_probability'] > 0.3 and result_low['arrhythmia_probability'] < 0.6:
        if result_low['prediction'] == 1 and result_high['prediction'] == 0:
            print("✅ Threshold logic working correctly!")
        else:
            print("❌ Threshold logic FAILED - predictions should differ")
    else:
        print("ℹ️  Signal not in threshold range for differential test")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    sensitivity_gain = sensitivity_threshold - sensitivity_baseline
    specificity_loss = specificity_baseline - specificity_threshold
    fn_reduction = fn - fn2

    print(f"\n📊 Metrics Comparison:")
    print(f"{'Metric':<20} {'Baseline':<15} {'Threshold=0.40':<15} {'Change'}")
    print("-" * 70)
    print(f"{'Sensitivity':<20} {sensitivity_baseline:>7.1%}{' '*7} {sensitivity_threshold:>7.1%}{' '*7} {sensitivity_gain:>+7.1%}")
    print(f"{'Specificity':<20} {specificity_baseline:>7.1%}{' '*7} {specificity_threshold:>7.1%}{' '*7} {-specificity_loss:>+7.1%}")
    print(f"{'False Negatives':<20} {fn:>7d}{' '*7} {fn2:>7d}{' '*7} {-fn_reduction:>+7d}")
    print(f"{'False Positives':<20} {fp:>7d}{' '*7} {fp2:>7d}{' '*7} {fp2-fp:>+7d}")

    print("\n🎯 Expected Behavior:")
    print("  ✓ Sensitivity should increase (fewer missed arrhythmias)")
    print("  ✓ Specificity should decrease (more false alarms)")
    print("  ✓ False negatives should decrease")
    print("  ✓ False positives should increase")

    print("\n🔍 Actual Results:")
    checks = []
    checks.append(("Sensitivity increased", sensitivity_gain > 0))
    checks.append(("Specificity decreased", specificity_loss > 0))
    checks.append(("False negatives decreased", fn_reduction > 0))
    checks.append(("False positives increased", (fp2 - fp) > 0))
    checks.append(("arrhythmia_probability present", len(arrhythmia_probs) > 0))

    all_passed = all(check[1] for check in checks)

    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ VALIDATION PASSED: Threshold implementation working correctly!")
    else:
        print("⚠️  VALIDATION ISSUES: Some checks failed")
    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = test_threshold_implementation()
    sys.exit(0 if success else 1)
