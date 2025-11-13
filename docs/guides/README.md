# Technical Guides

**Purpose**: Active how-to documentation for implementing CortexCore features

These guides are **living documents** - updated as implementation evolves and new patterns emerge.

---

## Available Guides

### Core SNN Development

#### `STDP_GUIDE.md` - Spike-Timing-Dependent Plasticity
**When to use**: Implementing biological learning rules, Phase 2+ enhancement

**Contains**:
- STDP theory and biological plausibility
- Complete snnTorch STDP implementation
- Hybrid STDP + backprop architecture
- Training loops and visualization
- Troubleshooting common issues

**Status**: ⭐ **CRITICAL** - Required by problem statement (PS.txt)

---

#### `CODE_EXAMPLES.md` - Common SNN Patterns
**When to use**: Daily development, debugging, implementing new features

**Contains**:
- Dual-output pattern (spikes + membrane)
- Time-first convention
- State initialization patterns
- Spike encoding examples
- Checkpoint loading/saving
- Common error fixes

**Status**: ⭐ **FREQUENTLY REFERENCED**

---

### Advanced Features

#### `ENSEMBLE_AVERAGING_GUIDE.md` - Variance Reduction
**When to use**: Reducing prediction variance from stochastic spike encoding

**Contains**:
- Why ensemble averaging is needed
- API usage (`ensemble_predict()`)
- Performance vs accuracy tradeoffs
- Clinical deployment recommendations
- Uncertainty quantification

**Achievements**: 59% variance reduction with N=5 ensemble

**Status**: ✅ **IMPLEMENTED** - Production-ready

---

#### `TRANSFER_LEARNING_SETUP.md` - Real Data Adaptation
**When to use**: Adapting synthetic-trained models to MIT-BIH or other real datasets

**Contains**:
- Data preprocessing pipelines
- Feature freezing strategies
- Fine-tuning hyperparameters
- Domain adaptation techniques
- Validation protocols

**Status**: 🚀 **ACTIVE** - Currently being implemented (Phase 8)

---

## Guide Usage Workflow

### For New Developers

1. **Start here**: `CODE_EXAMPLES.md` - Learn SNN patterns
2. **Understand learning**: `STDP_GUIDE.md` - Biological plausibility requirement
3. **Production features**: `ENSEMBLE_AVERAGING_GUIDE.md` - Variance reduction
4. **Real data**: `TRANSFER_LEARNING_SETUP.md` - MIT-BIH adaptation

### For Debugging

1. Check `CODE_EXAMPLES.md` for common error patterns
2. Verify spike encoding with examples from `ENSEMBLE_AVERAGING_GUIDE.md`
3. If STDP-related, see `STDP_GUIDE.md` troubleshooting section

### For Feature Implementation

1. Search existing guides for similar patterns
2. Reference `CODE_EXAMPLES.md` for SNN-specific gotchas
3. Document new patterns in `CODE_EXAMPLES.md` after validation

---

## Contributing to Guides

### When to Add Content

- New SNN pattern identified → Add to `CODE_EXAMPLES.md`
- STDP-related enhancement → Update `STDP_GUIDE.md`
- Ensemble method improvement → Update `ENSEMBLE_AVERAGING_GUIDE.md`
- Real data preprocessing step → Add to `TRANSFER_LEARNING_SETUP.md`

### Documentation Standards

- **Code examples**: Always include working, tested code
- **Explanations**: Focus on WHY, not just WHAT
- **Gotchas**: Document common errors and solutions
- **References**: Link to `src/` files for implementation
- **Status**: Mark sections as ✅ IMPLEMENTED, 🚧 IN PROGRESS, or 📋 PLANNED

---

## Related Documentation

- **Implementation Details**: `/docs/implementation/` - What was built
- **Results**: `/docs/results/` - Performance benchmarks
- **Decisions**: `/docs/decisions/` - Why technical choices were made
- **Planning**: `/docs/planning/` - What's next

---

## Quick Reference

| Need | Guide | Section |
|------|-------|---------|
| Fix state device mismatch | `CODE_EXAMPLES.md` | SNN Critical Gotchas |
| Implement STDP layer | `STDP_GUIDE.md` | Complete STDP SNN Class |
| Reduce prediction variance | `ENSEMBLE_AVERAGING_GUIDE.md` | API Usage |
| Load MIT-BIH data | `TRANSFER_LEARNING_SETUP.md` | Data Preprocessing |
| Dual-output pattern | `CODE_EXAMPLES.md` | Critical SNN Patterns |
