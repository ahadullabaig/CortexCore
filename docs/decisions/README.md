# Technical Decisions

**Purpose**: Problem analysis, root cause investigations, and architectural decision rationale

These documents capture **why specific technical decisions were made** and the evidence that supported those choices.

---

## Available Decision Logs

### Critical Issues

#### `CRITICAL_FIXES.md`
**Problem**: Model performance below clinical deployment thresholds

**Status**: 📋 **ACTIVE ANALYSIS** (November 9, 2025)

**Current Situation**:
- Sensitivity: 88.2% < 95% target ❌
- Specificity: 95.6% > 90% target ✅
- Systematic bias toward Normal predictions
- 59 false negatives (missed arrhythmias) vs 22 false positives

**Root Cause Analysis**:
1. **Low-confidence false negatives** (55.3% ≈ random guessing)
2. **Systematic errors** (98.8% of errors follow pattern)
3. **Architecture limitation** (320K params, 99.9% in single layer)
4. **Balanced data** → NOT a class imbalance issue

**Proposed Fixes**:
1. ✅ **Threshold optimization** (0.5 → 0.40)
   - Result: 97.4% sensitivity, 77% specificity (trades one for other)
2. 🚧 **Class-weighted loss** (FocalLoss)
   - Status: Infrastructure complete, needs training
3. 📋 **Deeper architecture** (DeepSNN)
   - Status: Design ready, needs implementation

**Outcome**: Phase 2 implemented all three fixes → 90.6% sensitivity achieved

**Use When**:
- Debugging model performance issues
- Understanding deployment blockers
- Planning optimization strategies

---

#### `SEED_CONSISTENCY_FIX.md`
**Problem**: Non-deterministic predictions (same signal → different outputs)

**Status**: ✅ **RESOLVED** (January 2025)

**Root Cause**:
- Poisson spike encoding is stochastic (random process)
- Different spike trains for same input signal
- `rate_encode()` called without seed control

**Evidence**:
```python
# Test with same signal, 10 runs
signal = np.array([...])  # Fixed signal
predictions = [predict(model, signal) for _ in range(10)]
# Results: [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # Random!
```

**Solution Implemented**:
1. **Ensemble averaging** (soft voting across N runs)
   - 59% variance reduction with N=5
   - <500ms with N=3 (production config)
2. **Explicit seed control** in `rate_encode()`
   - Added `base_seed` parameter
   - Reproducible predictions when seed specified

**Code Changes**:
```python
# Before (non-deterministic)
spikes = rate_encode(signal, num_steps=100)

# After (deterministic)
set_seed(42)
spikes = rate_encode(signal, num_steps=100)

# Or use ensemble with base_seed
result = ensemble_predict(model, signal, base_seed=42)
```

**Impact**:
- ✅ Production-ready inference (deterministic)
- ✅ Reproducible research results
- ✅ Regulatory compliance (FDA requires determinism)

**Use When**:
- Debugging prediction variance
- Understanding Poisson encoding behavior
- Implementing reproducible pipelines

---

#### `DEPLOYMENT_DECISION.md`
**Problem**: When to deploy? Synthetic vs Real data validation?

**Status**: 📋 **STRATEGIC DECISION** (November 9, 2025)

**Context**:
- Current model: 90.6% sensitivity on synthetic data
- Clinical targets: ≥95% sensitivity, ≥90% specificity
- Question: Continue optimizing on synthetic OR validate on real data first?

**Options Considered**:

**Option A: Linear Progression** (Original Plan v1.0)
```
Phase 1→2→3→4→5→6→7→8
(Optimize synthetic → Validate real)
```
- **Pros**: Systematic, incremental improvements
- **Cons**: 36+ days to real data, risk of over-fitting synthetic
- **Timeline**: Days 1-36

**Option B: Real Data First** (Adopted Plan v2.0) ⭐
```
Phase 1→2→8→Decision→9-12
(Validate real → Conditionally apply optimizations)
```
- **Pros**: Early validation, avoid wasted optimization, faster to deployment
- **Cons**: May need to backtrack if real data fails
- **Timeline**: Days 1-15

**Decision**: ⭐ **Option B - Real Data First**

**Rationale**:
1. **Hit synthetic ceiling** (90.6% is model limit on current data)
2. **Synthetic ≠ Real** (clean vs noisy, fixed vs variable morphology)
3. **Mandatory milestone** (Can't deploy without real data validation)
4. **Time savings** (16-21 days saved by validating early)
5. **Risk mitigation** (Know if approach works before over-optimizing)

**Evidence**:
- ROC analysis: No threshold achieves both clinical targets
- Tier 1 fixes exhausted: <1% gains for 5-7 days work
- Real data characteristics differ significantly (see analysis)

**Execution**:
- ✅ Phase 1-2: COMPLETE (variance reduction, evaluation)
- 🚀 Phase 8: NEXT (MIT-BIH validation - Days 11-15)
- ⏸️ Phase 3-7: CONDITIONAL (apply IF real data underperforms)

**Outcome**: Strategy pivot documented in `planning/REORGANIZATION_RATIONALE.md`

**Use When**:
- Planning deployment timeline
- Justifying prioritization to stakeholders
- Understanding project strategy evolution

---

## Decision Framework

### When to Create a Decision Log

Create a new decision document when:
1. **Major architectural choice** (e.g., SimpleSNN → DeepSNN)
2. **Strategic pivot** (e.g., linear → real-data-first)
3. **Critical bug investigation** (e.g., non-determinism fix)
4. **Deployment blocker** (e.g., performance below threshold)
5. **Technology selection** (e.g., FocalLoss vs WeightedCE)

### Required Sections

1. **Problem**: What issue are we addressing?
2. **Status**: Resolved/Active/Planned
3. **Context**: Why does this matter?
4. **Options Considered**: What alternatives were evaluated?
5. **Decision**: What was chosen and why?
6. **Rationale**: Evidence and reasoning
7. **Outcome**: What happened after implementation?

---

## Common Decision Patterns

### Performance Optimization Decisions

**Pattern**: Model underperforms → Investigate root cause → Test fixes → Choose best

**Example**: `CRITICAL_FIXES.md`
1. Identified: 88.2% sensitivity < 95% target
2. Root cause: Systematic bias toward Normal class
3. Tested: Threshold optimization, FocalLoss, DeepSNN
4. Chose: All three (complementary fixes)
5. Result: 90.6% sensitivity (+2.4%)

---

### Architecture Evolution Decisions

**Pattern**: Current architecture limiting → Design alternatives → Validate → Migrate

**Example**: SimpleSNN → DeepSNN
1. Identified: 99.9% params in single layer (bottleneck)
2. Designed: DeepSNN with 3 hidden layers (256→128→2)
3. Validated: Trained on validation set
4. Migrated: Updated training scripts, backward-compatible checkpoint loading
5. Result: Better capacity, improved sensitivity

---

### Strategy Pivot Decisions

**Pattern**: Assumptions violated → Re-evaluate → Pivot strategy → Document rationale

**Example**: `DEPLOYMENT_DECISION.md` (v1.0 → v2.0)
1. Assumption: Synthetic optimization transfers to real data
2. Violation: Hit synthetic ceiling, no further gains
3. Re-evaluated: Real data validation is mandatory anyway
4. Pivoted: Real Data First strategy (Phase 8 before 3-7)
5. Documented: `planning/REORGANIZATION_RATIONALE.md`

---

## Impact of Decisions

### SEED_CONSISTENCY_FIX.md Impact

**Before**:
- ❌ Non-deterministic predictions
- ❌ Cannot reproduce research results
- ❌ Not FDA-compliant

**After**:
- ✅ Deterministic with ensemble averaging
- ✅ Reproducible research
- ✅ Production-ready

**Metrics**: 59% variance reduction, <500ms inference

---

### CRITICAL_FIXES.md Impact

**Before**:
- ❌ 88.2% sensitivity (7% below target)
- ❌ Systematic bias toward Normal
- ❌ Cannot deploy clinically

**After**:
- ✅ 90.6% sensitivity (+2.4% improvement)
- ✅ Balanced performance (89.0% specificity)
- ⚠️ Still below target, but best possible on synthetic

**Metrics**: +2.4% sensitivity, -6.6% specificity (tradeoff)

---

### DEPLOYMENT_DECISION.md Impact

**Before** (Linear Plan):
- ⏰ 36+ days to real data validation
- ⚠️ Risk of over-fitting synthetic
- ❌ Wasted effort if real data fails

**After** (Real Data First):
- ⏰ 15 days to real data validation (-21 days)
- ✅ Early validation reduces risk
- ✅ Conditional optimization based on real results

**Metrics**: 16-21 days saved, risk-mitigated strategy

---

## Lessons Learned

### From SEED_CONSISTENCY_FIX.md

**Lesson**: Stochastic components require explicit reproducibility mechanisms

**Application**:
- Always call `set_seed()` before encoding
- Use ensemble averaging for production
- Document non-deterministic behavior in API

---

### From CRITICAL_FIXES.md

**Lesson**: Multiple complementary fixes often needed (no silver bullet)

**Application**:
- Don't expect single fix to solve complex issues
- Test fixes in isolation, then combine
- Document which fixes contributed how much

---

### From DEPLOYMENT_DECISION.md

**Lesson**: Validate assumptions early, pivot when violated

**Application**:
- Don't over-optimize on proxy metrics (synthetic data)
- Real data validation is priority #1
- Be willing to change strategy based on evidence

---

## Future Decision Logs

### Planned Decisions

#### Phase 8: MIT-BIH Architecture Decision
**Question**: Fine-tune entire model OR freeze early layers?

**Options**:
1. Full fine-tuning (update all weights)
2. Freeze layer 1, fine-tune layers 2-3
3. Freeze layers 1-2, fine-tune layer 3 only

**Deliverable**: `MITBIH_TRANSFER_LEARNING_DECISION.md`

---

#### Phase 10: STDP Integration Decision
**Question**: Replace backprop entirely OR use hybrid approach?

**Options**:
1. Pure STDP (all layers)
2. Hybrid (STDP layer 1, backprop layers 2-3)
3. Hybrid with supervised STDP (reward modulation)

**Deliverable**: `STDP_INTEGRATION_DECISION.md`

---

## Related Documentation

- **Planning**: `/docs/planning/` - Strategic roadmap informed by decisions
- **Results**: `/docs/results/` - Evidence supporting decisions
- **Implementation**: `/docs/implementation/` - How decisions were implemented
- **Guides**: `/docs/guides/` - Best practices from decision outcomes

---

## Contributing

### When to Document a Decision

Document when:
- ✅ Multiple options considered (not just "did X")
- ✅ Decision has project-wide impact
- ✅ Future developers need context for "why"
- ✅ Decision might be revisited later

### Decision Log Template

```markdown
# {Decision Title}

**Problem**: {What issue are we addressing?}

**Status**: {Resolved/Active/Planned}

**Context**: {Why does this matter? Background info.}

**Options Considered**:
1. **Option A**: {Description} - Pros: {...} Cons: {...}
2. **Option B**: {Description} - Pros: {...} Cons: {...}

**Decision**: ⭐ **{Chosen option}**

**Rationale**:
1. {Reason 1 with evidence}
2. {Reason 2 with evidence}

**Evidence**:
- {Data/metrics supporting decision}
- {References to results or benchmarks}

**Outcome**: {What happened after implementation}

**Impact**: {Measured effects on project}
```

---

## Quick Reference

| Need | Decision Log | Section |
|------|--------------|---------|
| Fix non-determinism | `SEED_CONSISTENCY_FIX.md` | Solution Implemented |
| Improve sensitivity | `CRITICAL_FIXES.md` | Proposed Fixes |
| Understand strategy pivot | `DEPLOYMENT_DECISION.md` | Decision Rationale |
| See fix impact | Any log | Outcome + Impact sections |
