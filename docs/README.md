# CortexCore Documentation

**Last Updated**: November 13, 2025

This directory contains all project documentation organized by purpose and lifecycle stage.

## Documentation Structure

```
docs/
├── guides/                # Active technical references (HOW-TO)
├── planning/              # Roadmap & strategic planning
├── results/               # Evaluation reports & benchmarks
├── implementation/        # Implementation summaries (WHAT was built)
└── decisions/             # Problem analysis & technical decisions
```

## Quick Navigation

### Getting Started
- **Implementation Guide**: See `guides/CODE_EXAMPLES.md` for common SNN patterns
- **STDP Learning**: See `guides/STDP_GUIDE.md` for biological learning implementation
- **Current Roadmap**: See `planning/ROADMAP_QUICK_REFERENCE.md` for development status

### Understanding Results
- **Latest Evaluation**: See `results/phase2/PHASE2_EVALUATION_REPORT.md`
- **Phase 1 Optimization**: See `results/phase1/TIER1_FINAL_RESULTS.md`
- **Ensemble Performance**: See `results/ensemble/ENSEMBLE_TESTING_REPORT.md`

### Feature Implementation
- **Ensemble Averaging**: See `implementation/ENSEMBLE_IMPLEMENTATION_SUMMARY.md`
- **Architecture Migration**: See `implementation/MIGRATION_SUMMARY.md`
- **Frontend Design**: See `implementation/FRONTEND_REDESIGN.md`

### Problem Analysis
- **Critical Fixes**: See `decisions/CRITICAL_FIXES.md` for deployment blockers
- **Seed Consistency**: See `decisions/SEED_CONSISTENCY_FIX.md` for reproducibility fixes

---

## Directory Descriptions

### `/guides/` - Technical References

**Purpose**: Active how-to documentation for implementing features

**Contains**:
- `STDP_GUIDE.md` - Spike-Timing-Dependent Plasticity implementation
- `CODE_EXAMPLES.md` - Common SNN coding patterns and snippets
- `ENSEMBLE_AVERAGING_GUIDE.md` - Variance reduction via ensemble methods
- `TRANSFER_LEARNING_SETUP.md` - Real data adaptation guide

**Use When**: Implementing new features or debugging existing code

---

### `/planning/` - Roadmap & Strategy

**Purpose**: Development roadmap, strategic decisions, and future planning

**Contains**:
- `ROADMAP_QUICK_REFERENCE.md` - Current active roadmap (Phase 1-12 status)
- `NEXT_STEPS_REORGANIZED.md` - Current development plan (v2.0 - Real Data First)
- `REORGANIZATION_RATIONALE.md` - Why we pivoted from linear to real-data-first
- `archived/NEXT_STEPS_DETAILED.md` - Superseded linear progression plan (v1.0)

**Use When**: Planning next sprint, understanding project priorities, or reviewing strategic decisions

---

### `/results/` - Evaluation Reports

**Purpose**: Benchmark results, evaluation metrics, and performance analysis

**Structure**:
```
results/
├── phase1/         # Tier 1 optimization (Days 1-7)
├── phase2/         # Comprehensive evaluation (Days 8-10)
└── ensemble/       # Ensemble method validation
```

**Contains**:
- **Phase 1**: Threshold optimization, architecture improvements (90.6% sensitivity achieved)
- **Phase 2**: Full test set evaluation, MIT-BIH preprocessing
- **Ensemble**: Variance reduction validation (59% variance reduction with N=5)

**Use When**: Analyzing model performance, tracking improvement over time, or preparing reports

---

### `/implementation/` - Implementation Summaries

**Purpose**: Documentation of WHAT was built and HOW it was integrated

**Contains**:
- `ENSEMBLE_IMPLEMENTATION_SUMMARY.md` - Ensemble averaging feature (variance reduction)
- `MIGRATION_SUMMARY.md` - Architecture evolution and code migrations
- `FRONTEND_REDESIGN.md` - UI/UX implementation details

**Use When**: Understanding feature implementation details, onboarding new developers, or reviewing architecture

---

### `/decisions/` - Technical Decisions

**Purpose**: Problem analysis, root cause investigations, and architectural decisions

**Contains**:
- `CRITICAL_FIXES.md` - Model deployment blockers and required fixes
- `SEED_CONSISTENCY_FIX.md` - Reproducibility issue resolution
- `DEPLOYMENT_DECISION.md` - Production deployment strategy

**Use When**: Understanding why specific technical decisions were made, debugging systemic issues

---

## Documentation Lifecycle

### Active Documents
**Updated regularly, referenced frequently**:
- `guides/*` - Living technical references
- `planning/ROADMAP_QUICK_REFERENCE.md` - Current roadmap
- `planning/NEXT_STEPS_REORGANIZED.md` - Current plan

### Historical Documents
**Preserved for reference, not actively updated**:
- `results/phase1/*` - Phase 1 benchmark results
- `results/phase2/*` - Phase 2 evaluation reports
- `planning/archived/*` - Superseded plans

### Decision Logs
**Written once, referenced when needed**:
- `decisions/*` - Technical decision rationale
- `implementation/*` - Implementation summaries

---

## Contributing to Documentation

### When Creating New Docs

1. **Choose the right directory**:
   - How-to guide? → `guides/`
   - Evaluation results? → `results/phaseN/`
   - Implementation summary? → `implementation/`
   - Technical decision? → `decisions/`
   - Planning/roadmap? → `planning/`

2. **Use consistent naming**:
   - UPPERCASE_WITH_UNDERSCORES.md
   - Descriptive names (avoid ambiguous abbreviations)

3. **Update this README**:
   - Add entry to Quick Navigation if high-priority
   - Update directory descriptions if adding new categories

### When Deprecating Docs

1. Move to `planning/archived/` if still useful for context
2. Add deprecation notice at the top of the file
3. Point to replacement document if applicable

---

## Related Documentation

- **Project Root**: `/CLAUDE.md` - Main project instructions for Claude Code
- **Context Files**: `/context/` - Enhanced and comprehensive development guides
- **Problem Statement**: `/context/PS.txt` - Original requirements

---

## Search Tips

**Find implementation patterns**:
```bash
grep -r "pattern_name" docs/guides/
```

**Find specific metrics**:
```bash
grep -r "sensitivity" docs/results/
```

**Find decision rationale**:
```bash
grep -r "why" docs/decisions/
```

**Find all TODOs**:
```bash
grep -r "TODO\|FIXME\|XXX" docs/
```
