# Tier 1 Fixes - Training Results Analysis

**Date:** 2025-11-09
**Model:** DeepSNN with FocalLoss (alpha=0.75, gamma=2.0)
**Training:** 50 epochs, early stopping at epoch 20
**Status:** ⚠️ Partial Success - Sensitivity target met, Specificity target not met

---

## Executive Summary

The DeepSNN model trained with FocalLoss achieved:
- ✅ **Sensitivity: 99.0%** (Target: ≥95%) - **EXCEEDS**
- ❌ **Specificity: 61.0%** (Target: ≥90%) - **29% below target**
- ✅ **NPV: 98.4%** (Target: ≥95%) - **PASS**
- ❌ **PPV: 71.7%** (Target: ≥85%) - **13.3% below**
- 🎯 **AUC-ROC: 0.9704** - **Excellent discrimination**

**Verdict:** Model has strong discrimination capability but FocalLoss alpha=0.75 is too aggressive, causing excessive false positives.

---

## Training Performance

### Best Model (Epoch 10)
```
Validation Set (1000 samples):
  Sensitivity:  96.8% ✅
  Specificity:  71.0% ❌
  Accuracy:     83.9%
  FN: 16 (false negatives)
  FP: 145 (false positives)
```

### Early Stopping
- Stopped at epoch 20 (10 epochs without improvement)
- Model oscillated between high sensitivity/low specificity configurations
- Could not find balance that met both targets simultaneously

---

## Test Set Performance

### Comprehensive Evaluation (Ensemble=3)
```
Test Set (1000 samples, 3-run ensemble averaging):

Clinical Metrics:
  Sensitivity:  99.0% ✅ (Target: ≥95%)
  Specificity:  61.0% ❌ (Target: ≥90%)
  Precision:    71.7% ❌ (Target: ≥85%)
  NPV:          98.4% ✅ (Target: ≥95%)
  AUC-ROC:      0.9704 ✅ (Excellent)
  Accuracy:     80.0%

Confusion Matrix:
                   Predicted
                Normal  Arrhythmia
  True Normal      305      195 (39% FPR)
       Arrhythmia    5      495 (1% FNR)
```

**Key Observations:**
1. **Excellent Arrhythmia Detection:** Only 5 missed out of 500 cases (99% sensitivity)
2. **Too Many False Alarms:** 195 out of 500 normal cases flagged (39% false positive rate)
3. **High AUC-ROC (0.9704):** Model has strong discrimination - threshold optimization should work
4. **Safety vs Alarm Fatigue Trade-off:** Very safe (few missed arrhythmias) but high alarm burden

---

## Root Cause Analysis

### Why Specificity is Low

**Primary Cause:** FocalLoss alpha=0.75 parameter

```python
criterion = FocalLoss(
    alpha=0.75,  # 75% weight to arrhythmia class
    gamma=2.0    # Focus on hard examples
)
```

**Impact:**
- Model learns to be **extremely** biased toward detecting arrhythmia
- Better to flag a normal case as arrhythmia than miss a true arrhythmia
- This creates the observed pattern: 99% sensitivity, 61% specificity

**Evidence:**
- Epoch 2: 96.4% sens, 58.2% spec (very aggressive from early on)
- Epoch 10: 96.8% sens, 71.0% spec (best balance found)
- Training never achieved >71% specificity

---

## Solution Pathways

### Option 1: Threshold Optimization (Quick, No Retraining)

**Approach:** Find classification threshold that balances sensitivity and specificity

**Feasibility Analysis:**
```
Current Operating Point (threshold ~0.5 with ensemble):
  Sensitivity: 99.0%
  Specificity: 61.0%

Desired Operating Point:
  Sensitivity: 95.0% (allow 4% reduction)
  Specificity: 90.0% (need 29% improvement)
```

**Can threshold optimization achieve this?**

With AUC-ROC = 0.9704, threshold optimization is VERY promising:
- High AUC means probability outputs are well-separated between classes
- We have "room" to reduce sensitivity (99% → 95%)
- Need to find threshold that trades 4% sensitivity for 29% specificity

**Recommendation:**
```python
# Try thresholds in range 0.55-0.70
# Higher threshold → lower sensitivity, higher specificity
# Expected sweet spot: ~0.60-0.65
```

**Pros:**
- No retraining required (saves ~45 minutes)
- Can test multiple thresholds quickly
- Preserves learned model weights

**Cons:**
- May not fully reach 90% specificity target
- Doesn't fix underlying model bias

**Estimated Success Probability:** 60-70%

---

### Option 2: Retrain with Adjusted FocalLoss Alpha (Recommended)

**Approach:** Retrain DeepSNN with lower alpha value

**Recommended Configuration:**
```bash
python scripts/train_tier1_fixes.py \
    --model deep \
    --epochs 50 \
    --alpha 0.60  # Reduced from 0.75
    --gamma 2.0 \
    --weight-decay 1e-4
```

**Rationale:**
- alpha=0.75 gives 75% weight to arrhythmia class (too aggressive)
- alpha=0.60 gives 60% weight to arrhythmia class (more balanced)
- Still prioritizes sensitivity but allows better specificity

**Expected Outcome:**
```
Sensitivity:  95-97% ✅ (slight reduction from 99%)
Specificity:  85-92% ✅ (significant improvement from 61%)
PPV:          85-90% ✅ (improved from 71.7%)
NPV:          95-97% ✅ (maintained)
```

**Pros:**
- Addresses root cause (model bias)
- More likely to achieve both targets
- Creates more balanced, deployable model

**Cons:**
- Requires retraining (~45-60 minutes)
- May need iteration if alpha=0.60 still not optimal

**Estimated Success Probability:** 80-85%

---

### Option 3: Hybrid Approach (Most Reliable)

**Step 1:** Retrain with alpha=0.60
**Step 2:** Fine-tune threshold on validation set
**Step 3:** Validate on test set

**Expected Outcome:**
```
After retraining with alpha=0.60:
  Base: Sensitivity ~96%, Specificity ~88%

After threshold optimization:
  Final: Sensitivity ~95%, Specificity ~90%+
```

**Estimated Success Probability:** 90-95%

---

## Comparison with Baseline

### Baseline Model (SimpleSNN, Standard Loss)
```
Test Set Performance:
  Sensitivity:  88.2%
  Specificity:  95.6%
  FN: 59
  FP: 22
```

### Current Model (DeepSNN, FocalLoss α=0.75)
```
Test Set Performance:
  Sensitivity:  99.0%  (+10.8%)
  Specificity:  61.0%  (-34.6%)
  FN: 5        (-54 fewer missed arrhythmias!)
  FP: 195      (+173 more false alarms)
```

**Analysis:**
- FocalLoss successfully increased sensitivity (88.2% → 99.0%)
- But over-corrected, sacrificing too much specificity
- **Sweet spot exists** between these two extremes

---

## Detailed Metrics Comparison

| Metric | Baseline | Fix #1 Only | DeepSNN α=0.75 | Target | Status |
|--------|----------|-------------|----------------|--------|--------|
| Sensitivity | 88.2% | 97.4% | **99.0%** | ≥95% | ✅ EXCEEDS |
| Specificity | 95.6% | 77.0% | **61.0%** | ≥90% | ❌ FAIL |
| PPV | 95.2% | ~80% | **71.7%** | ≥85% | ❌ FAIL |
| NPV | 89.0% | ~97% | **98.4%** | ≥95% | ✅ PASS |
| False Negatives | 59 | 13 | **5** | - | ✅ Excellent |
| False Positives | 22 | 115 | **195** | - | ❌ Too High |

---

## Recommendations

### Immediate Action (Next Steps)

**Recommended Path:** Option 2 (Retrain with alpha=0.60) + Option 1 (Threshold optimization)

**Step-by-Step Plan:**

#### Phase 1: Retrain with Optimized FocalLoss (Est. 60 min)
```bash
python scripts/train_tier1_fixes.py \
    --model deep \
    --epochs 50 \
    --alpha 0.60 \
    --gamma 2.0 \
    --weight-decay 1e-4
```

**Expected:** Sensitivity 95-97%, Specificity 85-92%

#### Phase 2: Evaluate on Test Set (Est. 5 min)
```bash
python scripts/comprehensive_evaluation.py \
    --model models/deep_focal_model.pt
```

#### Phase 3: Threshold Optimization if Needed (Est. 5 min)
```bash
python scripts/optimize_threshold.py \
    --model models/deep_focal_model.pt
```

#### Phase 4: Validate Final Configuration (Est. 5 min)
```bash
python scripts/validate_threshold_fix.py \
    --model models/deep_focal_model.pt \
    --threshold <optimal_threshold>
```

---

### Alternative: Quick Threshold Optimization First

If you want to test threshold optimization on the current model first:

```bash
# Create custom ROC analysis with ensemble predictions
# (Current script uses single predictions)
```

**Note:** Current ROC analysis uses single predictions (95.4% sens) while comprehensive evaluation uses ensemble=3 (99.0% sens). For production, we need threshold optimization on ensemble predictions.

---

## Success Criteria Checklist

### Must Pass (Deployment Ready)
- [ ] Sensitivity ≥ 95.0%
- [ ] Specificity ≥ 90.0%
- [ ] NPV ≥ 95.0%
- [ ] PPV ≥ 85.0%

### Current Status
| Criterion | Current | Target | Gap | Status |
|-----------|---------|--------|-----|--------|
| Sensitivity | 99.0% | ≥95% | +4.0% | ✅ EXCEEDS |
| Specificity | 61.0% | ≥90% | -29.0% | ❌ MAJOR GAP |
| PPV | 71.7% | ≥85% | -13.3% | ❌ MINOR GAP |
| NPV | 98.4% | ≥95% | +3.4% | ✅ EXCEEDS |

**Overall:** 2/4 targets met - Not deployment ready

---

## Technical Insights

### What Worked Well ✅

1. **DeepSNN Architecture:** 673K parameters with 3 layers successfully learned complex patterns
2. **FocalLoss Mechanism:** Effectively increased sensitivity by focusing on hard examples
3. **Early Stopping:** Prevented overfitting (training stopped appropriately at epoch 20)
4. **Regularization:** Dropout (0.3) + L2 weight decay (1e-4) maintained generalization
5. **Clinical Metrics Tracking:** Real-time monitoring enabled informed decisions

### What Needs Adjustment ⚠️

1. **FocalLoss Alpha:** 0.75 too aggressive → try 0.60 or 0.65
2. **Evaluation Consistency:** ROC analysis should use ensemble predictions for production relevance
3. **Threshold Consideration:** Training used argmax (0.5) but production may need different threshold

---

## Lessons Learned

### 1. FocalLoss Alpha Selection is Critical
- alpha=0.75 = 75% weight to minority class
- Small changes (0.75 → 0.60) can significantly impact balance
- **Lesson:** Start conservative (0.60), increase if needed

### 2. Ensemble Predictions Change Operating Point
- Single prediction: 95.4% sens, 77.6% spec
- Ensemble (N=3): 99.0% sens, 61.0% spec
- **Lesson:** Optimize threshold on same configuration used in production

### 3. High AUC-ROC Enables Threshold Optimization
- AUC-ROC = 0.9704 indicates strong class separation
- With this AUC, threshold optimization is highly viable
- **Lesson:** Check AUC-ROC before deciding to retrain vs threshold-tune

### 4. Early Stopping Worked Correctly
- Model couldn't find better balance after epoch 10
- Stopping at epoch 20 (10 patience) was appropriate
- **Lesson:** Early stopping prevents wasted compute when model plateaus

---

## Next Session Checklist

Before retraining:
- [ ] Decide: Threshold optimization first OR retrain immediately
- [ ] If retraining: Confirm alpha value (recommend 0.60 or 0.65)
- [ ] If threshold first: Update ROC script to use ensemble predictions
- [ ] Set clear stopping criteria (both targets met OR max iterations)

Files to update if needed:
- [ ] `scripts/optimize_threshold.py` - add ensemble prediction support
- [ ] `scripts/train_tier1_fixes.py` - already supports `--alpha` parameter

---

## Estimated Time to Completion

### Scenario A: Threshold Optimization Only
- Implement ensemble ROC analysis: 10 min
- Test thresholds: 5 min
- Validate: 5 min
- **Total: 20 minutes**
- **Success Probability: 60-70%**

### Scenario B: Retrain with Alpha=0.60
- Retrain (50 epochs): 45-60 min
- Evaluate: 5 min
- Threshold optimization (if needed): 10 min
- Validate: 5 min
- **Total: 65-80 minutes**
- **Success Probability: 80-85%**

### Scenario C: Retrain with Alpha=0.65 (Conservative)
- If alpha=0.60 over-corrects (too high specificity, low sensitivity)
- Same timeline as Scenario B
- **Success Probability: 85-90%**

---

## Final Recommendation

**RECOMMENDED APPROACH:** Retrain with alpha=0.60

**Rationale:**
1. Addresses root cause (model bias from alpha=0.75)
2. Higher success probability (80-85% vs 60-70%)
3. 65-80 minutes is acceptable for robust solution
4. Creates production-ready model vs temporary threshold fix
5. If successful, deployment ready immediately

**Command to Execute:**
```bash
python scripts/train_tier1_fixes.py \
    --model deep \
    --epochs 50 \
    --alpha 0.60 \
    --gamma 2.0 \
    --weight-decay 1e-4 \
    --device cuda
```

**Expected Outcome:**
- Sensitivity: 95-97% (meets target)
- Specificity: 85-92% (meets or approaches target)
- Training time: ~60 minutes
- If specificity ≥90%: Deployment ready ✅
- If specificity 85-89%: Apply threshold optimization to push over 90%

---

**Status:** Analysis complete, ready for decision
**Decision Needed:** Proceed with retraining (alpha=0.60) or try threshold optimization first?
