# Development Roadmap Quick Reference

**Last Updated**: November 9, 2025

---

## Phase Status Overview

| Phase | Name | Status | Days | Priority |
|-------|------|--------|------|----------|
| **1** | Stabilize Predictions | ✅ COMPLETE | 1-2 | CRITICAL |
| **2** | Comprehensive Evaluation | ✅ COMPLETE | 3-10 | HIGH |
| **8** | MIT-BIH Real Data | 🚀 **NEXT** | 11-15 | **CRITICAL** |
| **9** | Multi-Class Detection | 📋 Planned | 16-20 | HIGH |
| **10** | STDP Implementation | 📋 Planned | 21-27 | REQUIRED |
| **11** | Production Optimization | 📋 Planned | 28-30 | MEDIUM |
| **12** | Clinical Deployment | 🔄 Ongoing | 30+ | ONGOING |
| **3-7** | Synthetic Optimizations | ⏸️ DEFERRED | TBD | CONDITIONAL |

---

## Execution Sequence

### ✅ Completed (Days 1-10)

**Phase 1: Variance Reduction**
- Ensemble averaging (N=3)
- Seed consistency fix
- Deterministic predictions
- **Result**: Reproducible inference ✓

**Phase 2: Evaluation + Tier 1 Fixes**
- Full test set evaluation (1000 samples)
- FocalLoss + G-mean early stopping
- ROC threshold optimization
- DeepSNN architecture
- **Result**: 90.6% sens / 89.0% spec on synthetic

---

### 🚀 In Progress (Days 11-15)

**Phase 8: MIT-BIH Real Data Validation**

| Day | Task | Deliverable |
|-----|------|-------------|
| **11** | Data acquisition & preprocessing | `data/mitbih/{train,val,test}_ecg.pt` |
| **12** | Transfer learning setup | Training in progress |
| **13** | Training completion | Trained model checkpoint |
| **14** | Comprehensive evaluation | `docs/MITBIH_RESULTS.md` |
| **15** | Decision point | Next phase plan |

**Success Criteria**:
- ✅ Minimum: ≥85% sensitivity, ≥80% specificity
- ✅ Target: ≥90% sensitivity, ≥85% specificity
- ✅ Stretch: ≥95% sensitivity, ≥90% specificity

---

### 📋 Planned (Days 16-30)

**Phase 9: Multi-Class Detection** (if Phase 8 ≥90%)
- Expand to 5 classes (Normal, Atrial, Ventricular, Conduction, Paced)
- Train on same MIT-BIH data with multi-class labels
- **Target**: ≥85% accuracy across all classes

**Phase 10: STDP Implementation** (required)
- Hybrid STDP + Backprop learning
- Biological plausibility demonstration
- **Target**: ≥88% accuracy (within 4% of pure backprop)

**Phase 11: Production Optimization**
- Quantization (Float32 → Int8)
- Pruning (50-70% parameter reduction)
- Export (ONNX, TorchScript, TFLite)
- **Target**: <500KB model, <50ms inference on Jetson

**Phase 12: Clinical Deployment** (ongoing)
- Mobile app OR hospital integration
- Regulatory pathway (FDA 510k)
- Publication preparation

---

### ⏸️ Deferred (Conditional)

**Phase 3-7: Synthetic Optimizations**

These phases are DEFERRED until after Phase 8 results. Apply selectively based on real data performance:

| Phase | Tasks | When to Apply |
|-------|-------|---------------|
| **3** | Training improvements | If MIT-BIH <90% |
| **4** | Architecture enhancements | If MIT-BIH <88% |
| **5** | Encoding improvements | If MIT-BIH <85% |
| **6** | STDP | Always (after Phase 8) |
| **7** | Production | Always (after Phase 8) |

**Decision Logic**:
```
MIT-BIH Results ≥95%/90%: Skip Phase 3-5 → Go to Phase 9
MIT-BIH Results 90-94%/85-89%: Selective Phase 3-4 → Phase 9
MIT-BIH Results <90%/<85%: Full Phase 3-7 → Retry Phase 8
```

---

## Timeline Comparison

### Original Plan (Linear)
```
Days 1-2   | Phase 1 ✅
Days 3-4   | Phase 2 ✅
Days 5-11  | Phase 3 (synthetic training) ⏸️
Days 12-18 | Phase 4 (synthetic arch) ⏸️
Days 19-23 | Phase 5 (synthetic encoding) ⏸️
Days 24-30 | Phase 6 (STDP on synthetic) ⏸️
Days 31-35 | Phase 7 (production) ⏸️
Days 36-40 | Phase 8 (MIT-BIH) ← FIRST real data test!
```
**Total to real data**: 36 days

### New Plan (Real Data First)
```
Days 1-2   | Phase 1 ✅
Days 3-10  | Phase 2 + Tier 1 fixes ✅
Days 11-15 | Phase 8 (MIT-BIH) 🚀 ← Real data validation
Days 16-20 | Phase 9 (multi-class)
Days 21-27 | Phase 10 (STDP)
Days 28-30 | Phase 11 (production)
Days 30+   | Phase 12 (deployment)
```
**Total to real data**: 15 days (21 days saved ✓)

---

## Why We Reorganized

### 5 Key Reasons

1. **Hit Synthetic Ceiling**: ROC proves NO threshold achieves both targets (≥95% sens AND ≥90% spec)
2. **Synthetic ≠ Real**: Synthetic data doesn't capture patient variability, noise, artifacts
3. **Already Implemented Core Fixes**: DeepSNN + FocalLoss + G-mean = Phase 3-4 equivalents done
4. **Mandatory Validation**: Can't deploy/publish without real-world data anyway
5. **Evidence-Based Optimization**: Error analysis on real data guides targeted fixes (3x more efficient)

### What Changed

**Assumption Then** (Nov 7):
- SimpleSNN → Optimize on synthetic → Transfer to real
- Linear progression through all phases

**Reality Now** (Nov 9):
- DeepSNN + Tier 1 fixes → Hit model ceiling on synthetic
- Validate on real data FIRST, optimize IF needed

---

## Current Model Status

**Architecture**: DeepSNN (2500 → 256 → 128 → 2)
**Parameters**: 673,922
**Training**: FocalLoss(α=0.60, γ=2.0) + G-mean early stopping
**Inference**: Ensemble=3, threshold=0.577 (ROC-optimized)

**Performance (Synthetic Test Set, 1000 samples)**:
```
Sensitivity:  90.6% (Target: ≥95%, Gap: -4.4%)
Specificity:  89.0% (Target: ≥90%, Gap: -1.0%)
AUC-ROC:      0.9739 (Excellent)
Accuracy:     89.5%
PPV:          89.2%
NPV:          90.4%
```

**Confusion Matrix**:
```
                 Predicted
              Normal  Arrhythmia
True Normal      442        58
     Arrhythmia   47       453
```

---

## Next Immediate Steps (Day 11)

### Morning
1. [ ] Register PhysioNet account
2. [ ] Download MIT-BIH database (~100MB)
3. [ ] Install wfdb: `pip install wfdb`
4. [ ] Write `scripts/preprocess_mitbih.py`

### Afternoon
5. [ ] Run preprocessing (resample 360Hz→250Hz, filter, segment)
6. [ ] Patient-based split (34 train, 7 val, 7 test patients)
7. [ ] Save to `data/mitbih/` directory
8. [ ] Validate data quality (plot samples)

### Evening
9. [ ] Write `src/data.py:MITBIHDataset` class
10. [ ] Test data loader
11. [ ] Document in `docs/MITBIH_PREPROCESSING.md`

**End of Day 11**: Ready to start training

---

## Reference Documents

- **Full Reorganized Roadmap**: `docs/NEXT_STEPS_REORGANIZED.md` (40+ pages)
- **Reorganization Rationale**: `docs/REORGANIZATION_RATIONALE.md` (explains why)
- **Original Roadmap**: `docs/NEXT_STEPS_DETAILED.md` (archived, for reference)
- **Tier 1 Results**: `docs/TIER1_FINAL_RESULTS.md`
- **Deployment Decision**: `docs/DEPLOYMENT_DECISION.md`
- **Seed Consistency Fix**: `docs/SEED_CONSISTENCY_FIX.md`
- **Critical Fixes**: `docs/CRITICAL_FIXES.md`

---

## Quick Decision Matrix

### "Should I work on Phase X?"

| Phase | Work on it if... |
|-------|------------------|
| 1-2 | ✅ Already complete |
| 8 | ✅ Do this NOW (Day 11-15) |
| 9 | ✅ If Phase 8 achieves ≥90% |
| 10 | ✅ If Phase 8-9 complete (required) |
| 11 | ✅ If Phase 8-10 complete (required for deployment) |
| 12 | ✅ If Phase 8-11 complete (ongoing) |
| 3-7 | ⚠️ Only if Phase 8 <90% (selective) OR Phase 8 <85% (comprehensive) |

---

## Success Metrics Tracker

### Phase 1 ✅
- [x] Ensemble prediction variance < 0.10 (achieved: deterministic)
- [x] Reproducible results across runs
- [x] Documentation complete

### Phase 2 ✅
- [x] Test accuracy measured (89.5%)
- [x] Clinical metrics computed (sensitivity, specificity, etc.)
- [x] ROC threshold optimized (0.577)
- [x] Confusion matrix analyzed

### Phase 8 🚀 (Target)
- [ ] MIT-BIH preprocessing complete
- [ ] Transfer learning training complete
- [ ] Test accuracy ≥90%
- [ ] Sensitivity ≥90%, Specificity ≥85%
- [ ] Error analysis documented

### Phase 9 📋 (Target)
- [ ] 5-class detection implemented
- [ ] Per-class accuracy ≥85%
- [ ] Macro F1-score ≥0.88

### Phase 10 📋 (Required)
- [ ] STDP implementation complete
- [ ] Hybrid model accuracy ≥88%
- [ ] Biological plausibility documented

### Phase 11 📋 (Target)
- [ ] Quantized model <500KB
- [ ] Inference <50ms on Jetson
- [ ] ONNX/TorchScript/TFLite exports

### Phase 12 🔄 (Ongoing)
- [ ] Deployment demo (mobile OR hospital)
- [ ] Clinical validation initiated
- [ ] Publication draft complete

---

**Last Updated**: November 9, 2025
**Status**: Phase 8 starting
**Next Milestone**: MIT-BIH results (Day 15)
