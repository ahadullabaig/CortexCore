# Development Planning

**Purpose**: Strategic roadmap, development priorities, and planning decisions

This directory tracks the **evolution of project strategy** from initial plans through pivots and adjustments.

---

## Current Active Documents

### `ROADMAP_QUICK_REFERENCE.md`
**Status**: ⭐ **CURRENT ROADMAP** (Updated: November 9, 2025)

**Contains**:
- Phase status overview (Phases 1-12)
- Execution sequence and timeline
- Current priorities and next steps
- Completed vs planned work

**Use When**:
- Planning next sprint
- Checking project status
- Understanding current priorities

**Key Insights**:
- Phases 1-2: ✅ COMPLETE (variance reduction, evaluation)
- Phase 8: 🚀 **NEXT** (MIT-BIH real data validation)
- Phases 3-7: ⏸️ DEFERRED (synthetic optimizations conditional on real data)
- Phases 9-12: 📋 PLANNED (multi-class, STDP, production)

---

### `NEXT_STEPS_REORGANIZED.md`
**Status**: ⭐ **ACTIVE PLAN v2.0** (November 9, 2025)

**Contains**:
- Detailed phase-by-phase execution plan
- "Real Data First" strategy rationale
- Task breakdowns and deliverables
- Timeline estimates

**Use When**:
- Implementing current phase
- Understanding task dependencies
- Estimating completion time

**Strategic Pivot**: From linear progression (1→2→3→...→8) to **Real Data First** (1→2→**8**→conditional)

---

### `REORGANIZATION_RATIONALE.md`
**Status**: 📋 **DECISION LOG** (November 9, 2025)

**Contains**:
- Why we pivoted from v1.0 to v2.0 plan
- Evidence from Tier 1 optimization
- Synthetic vs real data analysis
- Risk mitigation strategy

**Use When**:
- Understanding why strategy changed
- Justifying prioritization decisions
- Learning from project evolution

**Key Decision**: Validate on real data (Days 11-15) BEFORE over-optimizing on synthetic (saves 16-21 days)

---

## Archived Documents

### `archived/NEXT_STEPS_DETAILED.md`
**Status**: 🗄️ **SUPERSEDED v1.0** (November 7, 2025)

**Original Plan**: Linear progression Phase 1→2→3→4→5→6→7→8

**Why Archived**:
- Assumed synthetic optimization transfers to real data (doesn't!)
- Hit synthetic data ceiling at 90.6% sensitivity
- ROC analysis proved no threshold achieves both clinical targets

**Still Useful For**:
- Understanding initial planning approach
- Comprehensive phase descriptions (Phases 3-7 detail)
- Historical context

**Replacement**: See `NEXT_STEPS_REORGANIZED.md` (v2.0)

---

## Timeline Evolution

### November 7, 2025 - v1.0 Plan
```
Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
(36+ days to reach real data validation)
```

**Assumption**: Optimize on synthetic, then validate on real

---

### November 9, 2025 - v2.0 Plan (Current)
```
Phase 1 → 2 → 8 → Decision → 9-12
(15 days to real data validation)
```

**Strategy**: Validate on real data early, then conditionally apply synthetic optimizations

---

## Using This Directory

### For Sprint Planning

1. **Check status**: `ROADMAP_QUICK_REFERENCE.md` - What phase are we in?
2. **Get tasks**: `NEXT_STEPS_REORGANIZED.md` - What specific work is next?
3. **Understand why**: `REORGANIZATION_RATIONALE.md` - Why this priority?

### For Stakeholder Updates

1. **Progress**: `ROADMAP_QUICK_REFERENCE.md` - Phase status table
2. **Strategy**: `NEXT_STEPS_REORGANIZED.md` - Current approach
3. **Decisions**: `REORGANIZATION_RATIONALE.md` - Strategic rationale

### For Future Planning

1. **Current work**: Phases 1-2 (complete), Phase 8 (in progress)
2. **Conditional work**: Phases 3-7 (apply IF real data underperforms)
3. **Upcoming work**: Phases 9-12 (multi-class, STDP, production)

---

## Phase Overview (from ROADMAP_QUICK_REFERENCE.md)

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

## Decision Framework

When planning work, ask:

1. **Does this validate on real data?** → High priority (Phase 8)
2. **Does this enable deployment?** → High priority (Phases 9-12)
3. **Does this optimize synthetic?** → Deferred until real data validates (Phases 3-7)
4. **Is this required by PS.txt?** → Must implement (STDP - Phase 10)

---

## Related Documentation

- **Results**: `/docs/results/` - Evidence for planning decisions
- **Implementation**: `/docs/implementation/` - What was actually built
- **Decisions**: `/docs/decisions/` - Technical decision rationale
- **Guides**: `/docs/guides/` - How to implement planned features

---

## Contributing to Planning

### When to Update

- **ROADMAP_QUICK_REFERENCE.md**: When phase status changes or priorities shift
- **NEXT_STEPS_REORGANIZED.md**: When tasks are completed or plans adjust
- **New decision log**: When major strategic pivot occurs

### Archive Strategy

- Move superseded plans to `archived/`
- Add deprecation notice at top of archived file
- Point to replacement document
- Keep for historical context (don't delete!)

---

## Key Lessons Learned

From v1.0 → v2.0 pivot:

1. **Validate assumptions early**: Don't over-optimize before testing on target data
2. **ROC analysis reveals limits**: 90.6% sensitivity was model ceiling, not threshold issue
3. **Synthetic ≠ Real**: Perfect synthetic performance doesn't guarantee real-world success
4. **Prioritize mandatory milestones**: Real data validation required for deployment
5. **Be willing to pivot**: Saved 16-21 days by reorganizing strategy
