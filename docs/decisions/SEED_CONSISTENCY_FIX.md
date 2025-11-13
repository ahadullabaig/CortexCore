# Seed Consistency Fix - Critical Reproducibility Issue

**Date**: November 9, 2025
**Issue**: Massive performance discrepancy due to random seed inconsistency
**Status**: ✅ **FIXED**

---

## Problem Discovery

During ROC threshold optimization with ensemble=3, we discovered a **critical reproducibility issue**:

### Observed Discrepancy

| Script | Ensemble Size | Sensitivity | Specificity | Result |
|--------|--------------|-------------|-------------|--------|
| **comprehensive_evaluation.py** | 3 | 90.4% | 90.4% | Balanced |
| **optimize_threshold.py** (before fix) | 3 | 99.4% | 66.8% | Aggressive |

**Same model, same ensemble size, completely different results!**

### Root Cause

Different scripts used **incompatible random seeding strategies** for stochastic spike encoding:

#### Before Fix

1. **comprehensive_evaluation.py**:
   ```python
   ensemble_predict(model, signal, ensemble_size=3)
   # No base_seed passed → defaults to None
   # Result: Non-deterministic, random spike encoding
   ```

2. **optimize_threshold.py** (original):
   ```python
   seed = 42 + ensemble_idx * 1000 + i  # Different pattern!
   # Sample 0: seeds [42, 1042, 2042]
   # Sample 1: seeds [43, 1043, 2043]
   ```

**Different seeds → Different spike trains → Completely different predictions**

---

## Solution: Unified Seeding Strategy

### Standard Seed Pattern (Now Used Everywhere)

For sample `i` and ensemble member `j`:
```python
seed = (42 + i * 1000) + j
```

**Examples**:
- Sample 0, ensemble members: `42, 43, 44`
- Sample 1, ensemble members: `1042, 1043, 1044`
- Sample 2, ensemble members: `2042, 2043, 2044`

This pattern ensures:
- ✅ **Deterministic**: Same input always produces same output
- ✅ **Unique**: Each (sample, ensemble_member) pair gets unique seed
- ✅ **Consistent**: All scripts use identical seeding
- ✅ **Reproducible**: Results can be verified across runs

---

## Files Modified

### 1. `scripts/comprehensive_evaluation.py` (Line 153)

**Before**:
```python
result = ensemble_predict(
    model=model,
    input_data=signal,
    ensemble_size=ensemble_size,
    device=str(device),
    return_confidence=True
)
```

**After**:
```python
# Use deterministic seed for reproducibility
sample_base_seed = 42 + i * 1000

result = ensemble_predict(
    model=model,
    input_data=signal,
    ensemble_size=ensemble_size,
    device=str(device),
    return_confidence=True,
    base_seed=sample_base_seed  # CRITICAL FIX
)
```

### 2. `scripts/optimize_threshold.py` (Lines 100-105)

**Before**:
```python
seed = 42 + ensemble_idx * 1000 + i  # Wrong pattern
```

**After**:
```python
# Use deterministic seed matching ensemble_predict() pattern
sample_base_seed = 42 + i * 1000
seed = sample_base_seed + ensemble_idx
set_seed(seed)
```

### 3. `demo/app.py` (Line 250)

**Status**: ✅ Already correct (passes `base_seed=seed`)

For demo API, uses fixed seed (42) or None based on `use_seed` parameter. This is acceptable for single-sample interactive predictions.

---

## Verification Required

After this fix, **comprehensive_evaluation.py must be re-run** to get true representative results.

### Critical Question to Answer

With consistent seeding, will comprehensive evaluation now show:
- **Same as before**: 90.4% sens / 90.4% spec (seeds didn't matter much)
- **Same as ROC**: 99.4% sens / 66.8% spec (previous result was lucky)
- **Something else**: New third result

### Re-run Command

```bash
source venv/bin/activate && python scripts/comprehensive_evaluation.py \
    --model models/deep_focal_model.pt
```

Expected output: **Deterministic, reproducible test set results**

---

## Why This Matters

### 1. Reproducibility Crisis Avoided

Without consistent seeding:
- Results vary wildly between runs
- Can't verify model performance
- Can't trust deployment metrics
- Violates FDA requirements for medical devices

### 2. Fair Comparison

All evaluation methods now use:
- Same spike encoding process
- Same random seeds
- Same operating point
- Comparable results

### 3. Deployment Confidence

With consistent seeding:
- Production performance matches evaluation
- Test set results are trustworthy
- Clinical validation is meaningful
- Model behavior is predictable

---

## Technical Background

### Why SNNs Need Deterministic Seeding

Unlike ANNs, SNNs use **stochastic Poisson spike encoding**:

```python
# Each encoding uses random sampling
signal_norm = normalize(signal)
spikes = np.random.rand(T, N) < (signal_norm * gain / 100.0)
```

**Problem**: Same signal → Different spike trains → Different predictions

**Solution**: Set random seed before each encoding

### Ensemble Averaging

Ensemble predictions average probabilities across N encodings:
```python
probs = []
for j in range(ensemble_size):
    seed = base_seed + j  # Each member uses different seed
    set_seed(seed)
    spikes = encode(signal)
    prob = model(spikes)
    probs.append(prob)

final_prob = mean(probs)  # Reduces variance
```

**Critical**: `base_seed` must be deterministic and sample-specific!

---

## Impact on Previous Results

### Results to Discard

All previous ensemble=3 evaluations with non-deterministic seeds are **invalid**:
- ❌ Previous comprehensive_evaluation.py results (pre-fix)
- ❌ Any comparison between scripts (used different seeds)

### Results Still Valid

- ✅ Training results (not affected by inference seeding)
- ✅ Model checkpoints (architecture unchanged)
- ✅ Single predictions with explicit seeds

---

## Next Steps

1. **Re-run comprehensive evaluation** with fixed seeding
2. **Compare results** with ROC optimization (should now match)
3. **Update deployment documentation** with final verified metrics
4. **Proceed with deployment decision** based on consistent results

---

## Lessons Learned

### For SNN Development

1. **Always use explicit seeds** for stochastic operations
2. **Document seeding strategy** in function docstrings
3. **Verify consistency** across all evaluation scripts
4. **Test reproducibility** before deployment

### For Machine Learning Research

1. **Non-determinism is a silent bug** - results look plausible but vary
2. **Ensemble methods need careful seeding** - each member must be reproducible
3. **Always verify metrics** across different evaluation methods
4. **Reproducibility is not optional** - especially for medical AI

---

**Critical Action Required**: Re-run comprehensive evaluation to get true model performance with consistent seeding.
