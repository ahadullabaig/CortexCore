# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Neuromorphic SNN for Healthcare Signal Pattern Recognition** (ECG/EEG analysis)

**Goals:**
- 60%+ energy efficiency vs CNNs
- 92%+ accuracy on disease detection
- <50ms inference time
- **CRITICAL**: STDP implementation required for biological plausibility (see `docs/guides/STDP_GUIDE.md`)

**Current Status (November 2025):**
- ✅ SNN simulation on GPU (snnTorch)
- ✅ Synthetic ECG data generation with realistic overlap
- ✅ DeepSNN architecture (673K params) with FocalLoss
- ✅ 89.5% test accuracy (90.6% sensitivity / 88.4% specificity)
- ✅ Phase 2 comprehensive evaluation complete
- ✅ MIT-BIH preprocessing complete (2,190 segments ready)
- 🔄 Real data validation in progress (transfer learning)
- ⚠️ STDP: Hybrid implementation available, backprop currently primary

**Success Metrics:**
- Synthetic Data (Achieved): 89.5% accuracy, 60% energy efficiency, <90ms inference
- MIT-BIH Target: 85-90% accuracy on real patient ECG data
- Clinical Targets: ≥95% sensitivity, ≥90% specificity (close, within 5%)

## Quick Start

```bash
# Setup
make install              # or: bash scripts/01_setup_environment.sh
source venv/bin/activate

# Data generation
make generate-data        # or: bash scripts/02_generate_mvp_data.sh

# Training
make train                # or: bash scripts/03_train_mvp_model.sh
make train-fast           # Quick 5-epoch test

# Demo
make demo                 # or: bash scripts/04_run_demo.sh (http://localhost:5000)

# Testing
make test                 # or: bash scripts/05_test_integration.sh

# Development
make notebook             # Launch Jupyter
make format               # Black + isort
make lint                 # Flake8
make clean                # Remove temp files
```

## Architecture

### Core Module Responsibilities

| File | Purpose | Key APIs |
|------|---------|----------|
| `src/data.py` | Data generation & spike encoding | `generate_synthetic_ecg()`, `rate_encode()`, `ECGDataset` |
| `src/model.py` | SNN architectures | `SimpleSNN`, `DeepSNN`, `STDPLayer`, `measure_energy_efficiency()` |
| `src/losses.py` | Loss functions | `FocalLoss`, `WeightedBCELoss`, `BalancedCELoss` |
| `src/train.py` | Training pipeline | `train_epoch()`, `validate()`, `train_model()` (backprop) |
| `src/train_stdp.py` | STDP training pipeline | `train_stdp_epoch()`, `train_hybrid()` (biological learning) |
| `src/stdp.py` | STDP algorithms | `stdp_update()`, `compute_stdp_weight_change()` |
| `src/inference.py` | Model loading & prediction | `load_model()`, `predict()`, `ensemble_predict()` |
| `src/preprocessing.py` | Real data preprocessing | MIT-BIH preprocessing, quality control, segmentation |
| `src/utils.py` | Utilities | `set_seed()`, `get_device()`, `calculate_clinical_metrics()` |
| `src/visualization.py` | Plotting & analysis | Spike rasters, ROC curves, confusion matrices |
| `demo/app.py` | Flask web interface | Routes: `/`, `/health`, `/api/predict`, `/api/visualize_spikes` |

### Pipeline Flow

```
Data Generation (src/data.py)
    ↓
Spike Encoding (rate_encode: Poisson process)
    ↓
SNN Forward Pass (src/model.py: LIF neurons)
    ↓
Training (src/train.py: surrogate gradients OR STDP)
    ↓
Model Checkpoint (models/best_model.pt)
    ↓
Inference (src/inference.py)
    ↓
Demo UI (demo/app.py: Flask server)
```

**Key Architecture Insight**: The project uses a two-phase learning strategy:
- **Phase 1 (MVP)**: Pure surrogate gradient backpropagation for fast convergence
- **Phase 2 (Enhancement)**: Hybrid STDP (layer 1) + backprop (layer 2) for biological plausibility

### Critical SNN Patterns

**1. Dual-Output Pattern**
```python
# Models return TWO tensors: (spikes, membrane)
spikes, membrane = model(x)  # Both: [time_steps, batch, classes]
output = spikes.sum(dim=0)   # Sum over time for loss
loss = criterion(output, target)
```

**2. Time-First Convention**
```python
# SNNs expect: [time_steps, batch, features]
# NOT: [batch, time_steps, features]
# Models auto-transpose if needed, but encode data correctly
```

**3. State Initialization (CRITICAL)**
```python
# MUST initialize neuron state before forward pass
mem = self.lif.init_leaky()  # Required! Forgetting causes errors
spk, mem = self.lif(cur, mem)
```

**4. Stochastic Spike Encoding**
```python
# rate_encode() uses Poisson process - same input = different spikes
# For reproducibility: set_seed() BEFORE encoding
from src.utils import set_seed
set_seed(42)
spikes = rate_encode(signal, num_steps=100, gain=10.0)
```

**5. Surrogate Gradients**
```python
# Spikes are binary (non-differentiable)
# Use surrogate gradients for backprop
spike_grad = surrogate.fast_sigmoid()
lif = snn.Leaky(beta=0.9, spike_grad=spike_grad)
```

### snnTorch Critical Gotchas

**State Device Mismatch** (most common error)
```python
model.to(device)
mem = lif.init_leaky().to(device)  # State MUST match model device!
# Symptom: "Expected all tensors to be on the same device"
```

**State Initialization** (second most common)
```python
# MUST call before EVERY forward pass (not just once)
def forward(self, x):
    mem1 = self.lif1.init_leaky()  # Initialize here!
    for t in range(x.size(0)):
        cur = self.fc1(x[t])
        spk, mem1 = self.lif1(cur, mem1)
```

**No Time Parallelization**
```python
# Must iterate sequentially over time dimension
for step in range(x.size(0)):  # Cannot parallelize across time
    cur = self.fc1(x[step])
    spk, mem = self.lif1(cur, mem)
```

**Checkpoint Format**
```python
# Training saves full context, not just weights
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'val_loss': val_loss
}
# inference.py:load_model() handles BOTH checkpoint dict and raw state_dict
```

**Reproducibility Trap**
```python
# rate_encode() is stochastic - call set_seed() BEFORE each encoding
# Otherwise: same input signal → different spike trains every time
from src.utils import set_seed
set_seed(42)  # Must be called before EACH encoding for reproducibility
spikes = rate_encode(signal, num_steps=100)
```

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `models/best_model.pt` | Trained model location |
| `DEVICE` | auto-detect | `cuda`, `cpu`, or `mps` |
| `BATCH_SIZE` | `32` | Training batch size (reduce for low VRAM) |
| `LEARNING_RATE` | `0.001` | Optimizer learning rate |
| `NUM_EPOCHS` | `50` | Training epochs |
| `SAMPLING_RATE` | `250` | ECG/EEG sampling rate (Hz) |
| `SIGNAL_DURATION` | `10` | Signal length (seconds) |

### Key Hardcoded Values

**Data Generation** (`scripts/02_generate_mvp_data.sh`):
- Normal ECG: 70 BPM, noise 0.05
- Arrhythmia: 120 BPM, noise 0.1
- Output: `train_ecg.pt`, `val_ecg.pt`, `test_ecg.pt`

**Model Architectures** (`src/model.py`):

**SimpleSNN (Baseline, 320K params)**:
- Input: 2500 (10s × 250Hz)
- Layer 1: FC(2500 → 128) + LIF
- Layer 2: FC(128 → 2) + LIF
- Beta (membrane decay): 0.9

**DeepSNN (Current, 673K params)**:
- Input: 2500 (10s × 250Hz)
- Layer 1: FC(2500 → 256) + LIF
- Layer 2: FC(256 → 128) + Dropout(0.3) + LIF
- Layer 3: FC(128 → 2) + LIF
- Beta (membrane decay): 0.9
- Training: FocalLoss(alpha=0.60, gamma=2.0) + G-mean early stopping

**Spike Encoding** (`src/data.py:rate_encode`):
- Default time steps: 100
- Default gain: 10.0 (range: 1.0-20.0)
- Method: Poisson process (stochastic)

**SNN Parameters**:
- `beta`: Membrane decay rate (0-1). Higher = longer memory
- `threshold`: Spike threshold (default 1.0). Lower = more spikes
- `spike_grad`: Surrogate gradient function (fast_sigmoid, sigmoid, atan, triangular)

## Development Workflow

### Script Dependencies (Run in Order)

```
01_setup_environment.sh       # BLOCKS all (creates venv, installs deps)
    ↓
02_generate_mvp_data.sh       # BLOCKS training (creates dataset)
    ↓
03_train_mvp_model.sh         # BLOCKS demo (trains model, saves best_model.pt)
    ↓
04_run_demo.sh                # Launches Flask at localhost:5000

05_test_integration.sh        # Independent (8 test suites)
```

**Expected Outputs:**
- After setup: `venv/`, `data/`, `models/`, `results/` directories
- After data: `data/synthetic/{train,val,test}_ecg.pt`, `mvp_dataset.pt`
- After training: `models/best_model.pt`, `results/metrics/training_history.json`
- After demo: Flask server on port 5000

### Utility Scripts (scripts/)

**Training & Optimization**:
- `train_full_stdp.py` - Full STDP training script (3-phase: STDP → Hybrid → Fine-tuning)
- `train_tier1_fixes.py` - Tier 1 optimization (FocalLoss + G-mean early stopping)
- `train_mitbih_transfer.py` - MIT-BIH transfer learning (2-stage pipeline)
- `preprocess_mitbih.py` - MIT-BIH preprocessing pipeline (quality control, segmentation)
- `optimize_threshold.py` - ROC curve threshold optimization

**Evaluation & Analysis**:
- `comprehensive_evaluation.py` - Phase 2 full evaluation suite (5 tasks, 1000 samples)
- `benchmark_stdp.py` - STDP performance benchmarks
- `analyze_dataset_quality.py` - Validate dataset distribution, check for class imbalance
- `evaluate_test_set.py` - Clinical metrics evaluation

**Testing & Validation**:
- `comprehensive_verification.py` - Full pipeline verification from data to inference
- `validate_ensemble_averaging.py` - Ensemble averaging validation
- `validate_threshold_fix.py` - Threshold optimization validation
- `validate_architectures.py` - Model architecture validation
- `test_inference.py` - Test model loading and prediction functions
- `test_flask_demo.py` - Test Flask endpoints and API responses

**Debugging**:
- `debug_model.py` - Model debugging diagnostics
- `quick_stdp_test.py` - Quick STDP functionality test

### Development Philosophy

**Hackathon MVP Approach** (from `context/ENHANCED_STRUCTURE.md`):
- ✅ Notebooks first, refactor later
- ✅ Single file modules initially
- ✅ Demo folder is top priority
- ✅ Hardcode first, configure later
- ✅ Working > perfect
- ❌ No premature abstraction
- ❌ No deep nesting
- ❌ No over-engineering

**Timeline:**
- Days 1-7: MVP (85% accuracy, working demo)
- Days 8-14: Enhancement (92% accuracy, STDP hybrid)
- Days 15-30: Production (edge deployment, mobile)

### Notebooks

1. `01_quick_prototype.ipynb` - Initial SNN concepts
2. `02_data_generation.ipynb` - Data exploration, spike encoding
3. `03_snn_training.ipynb` - Hyperparameter tuning
4. `04_demo_prep.ipynb` - Visualizations for demo

**Workflow**: Prototype in notebooks → Move working code to `src/` → Integrate with scripts → Deploy to demo

## Common Development Patterns

### Adding a New Disease Class

1. Update `src/data.py:generate_synthetic_ecg()` with new condition parameters
2. Modify `scripts/02_generate_mvp_data.sh` to generate new class data
3. Update `src/model.py:SimpleSNN` output_size if needed
4. Retrain model with `make train`
5. Update `demo/app.py` class labels for display

### Debugging Training Issues

**Check data quality first**:
```bash
python scripts/analyze_dataset_quality.py
# Look for: class imbalance, signal quality, spike encoding statistics
```

**Profile inference performance**:
```python
from src.inference import profile_inference
times, memory = profile_inference(model, test_signal, device='cuda')
```

**Visualize spike patterns**:
```bash
# Use demo endpoint to see neuron firing patterns
curl -X POST http://localhost:5000/api/visualize_spikes -d '{"signal": [...]}'
```

### Testing Individual Components

```bash
# Test data generation only
python -c "from src.data import generate_synthetic_ecg; print(generate_synthetic_ecg(n_samples=10).shape)"

# Test model forward pass
python -c "from src.model import SimpleSNN; import torch; m = SimpleSNN(); x = torch.randn(100, 1, 2500); print(m(x)[0].shape)"

# Test spike encoding
python -c "from src.data import rate_encode; import numpy as np; s = np.random.rand(2500); print(rate_encode(s, num_steps=100).shape)"

# Verify model checkpoint
python scripts/test_inference.py
```

## Common Issues

### Memory Issues

**CUDA Out of Memory**
```bash
# Quick fix: reduce batch size
BATCH_SIZE=16 python src/train.py  # Try 16, then 8 if still failing
```
- SNNs use more memory than ANNs due to sequential time step processing
- Memory usage: `batch_size × time_steps × hidden_size × 4 bytes`
- Reduce `num_steps` in spike encoding (100 → 50) as last resort

**CPU Memory Issues**
```bash
# Monitor memory during training
watch -n 1 'free -h'
# Reduce dataset size in scripts/02_generate_mvp_data.sh if needed
```

### Training Issues

**Model Not Converging (loss not decreasing)**
1. Check data quality: `python scripts/analyze_dataset_quality.py`
2. Verify spike encoding is working:
   ```python
   # Should see mix of 0s and 1s, not all 0s or all 1s
   from src.data import rate_encode
   spikes = rate_encode(signal, num_steps=100, gain=10.0)
   print(f"Spike rate: {spikes.mean():.2f}")  # Should be 0.05-0.30
   ```
3. Adjust learning rate: try `0.01` or `0.0001` instead of `0.001`
4. Change surrogate gradient: `fast_sigmoid` → `sigmoid` or `atan`

**Spike Encoding Produces All 0s or All 1s**
```python
# All 1s: gain too high
spikes = rate_encode(signal, num_steps=100, gain=5.0)  # Reduce from 10.0

# All 0s: gain too low OR signal not normalized
signal = (signal - signal.min()) / (signal.max() - signal.min())  # Normalize to [0,1]
spikes = rate_encode(signal, num_steps=100, gain=15.0)  # Increase from 10.0
```

**Non-Reproducible Results**
```python
# rate_encode() uses Poisson process (stochastic)
# MUST call set_seed() before EACH encoding
from src.utils import set_seed
set_seed(42)
spikes = rate_encode(signal)  # Now reproducible
```

### Model Loading Issues

**RuntimeError: Expected all tensors to be on the same device**
```python
# Solution 1: Initialize state on correct device
model.to(device)
mem = lif.init_leaky().to(device)  # Don't forget .to(device)!

# Solution 2: Move everything at once
model = model.to(device)
x = x.to(device)
mem = lif.init_leaky().to(device)
```

**State Initialization Error**
```python
# WRONG: Initializing outside forward pass
mem = self.lif.init_leaky()  # Only called once
def forward(self, x):
    spk, mem = self.lif(cur, mem)  # mem may be stale

# CORRECT: Initialize inside forward pass
def forward(self, x):
    mem = self.lif.init_leaky()  # Fresh state every forward pass
    spk, mem = self.lif(cur, mem)
```

**Checkpoint Loading Fails**
```python
# inference.py:load_model() handles both formats automatically
# If still fails, check manually:
checkpoint = torch.load(model_path)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```

### Environment Issues

**Import Errors After Installation**
```bash
# 1. Verify venv is activated
which python  # Should show venv path, not system python

# 2. Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# 3. For CUDA issues (Linux/Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Demo Not Loading / Returns Mock Predictions**
- Check model exists: `ls -lh models/best_model.pt`
- Verify Flask is running: `curl http://localhost:5000/health`
- Mock predictions are intentional for MVP (TODO: integrate real SNN)

**Scripts Fail with "No such file or directory"**
```bash
# Always run scripts from project root
cd /path/to/cortexcore
bash scripts/01_setup_environment.sh  # ✓ Correct
cd scripts && bash 01_setup_environment.sh  # ✗ Wrong
```

## Context & Documentation

### Current Development Priority (November 2025)

**⭐ NEXT MILESTONE: MIT-BIH Real Data Validation**

After completing Phase 2 evaluation and Tier 1 optimization, the project has shifted to a **Real Data First** strategy:

1. **Why the Pivot**:
   - Tier 1 optimization hit fundamental synthetic data limits
   - ROC analysis shows no threshold achieves both clinical targets
   - Real-world validation is mandatory for deployment
   - Day 10/30 - efficient to validate early vs over-optimizing synthetic

2. **What's Ready**:
   - ✅ MIT-BIH preprocessing complete: 2,190 high-quality ECG segments
   - ✅ Transfer learning pipeline implemented (2-stage training)
   - ✅ DeepSNN baseline model (89.5% accuracy on synthetic)
   - ✅ Evaluation infrastructure (comprehensive_evaluation.py)

3. **Next Steps** (See `docs/planning/NEXT_STEPS_REORGANIZED.md`):
   ```bash
   # Stage 1: Freeze layer 1, fine-tune layer 2 (20 epochs)
   python scripts/train_mitbih_transfer.py --stage 1

   # Stage 2: Full fine-tuning (30 epochs)
   python scripts/train_mitbih_transfer.py --stage 2

   # Or run both stages together
   python scripts/train_mitbih_transfer.py --stage both
   ```

4. **Success Criteria**: 85-90% accuracy on real patient ECG (MIT-BIH test set)

5. **Conditional Optimization**: Apply Phase 3-7 synthetic improvements ONLY if MIT-BIH underperforms

### Documentation Structure

Documentation is organized in `/docs/` by purpose and lifecycle:

```
docs/
├── README.md                      # Documentation overview & navigation
├── guides/                        # Active technical references (HOW-TO)
│   ├── STDP_GUIDE.md
│   ├── CODE_EXAMPLES.md
│   ├── ENSEMBLE_AVERAGING_GUIDE.md
│   └── TRANSFER_LEARNING_SETUP.md
├── planning/                      # Roadmap & strategic planning
│   ├── ROADMAP_QUICK_REFERENCE.md # Current active roadmap
│   ├── NEXT_STEPS_REORGANIZED.md  # Current plan (v2.0)
│   └── archived/                  # Superseded plans
├── results/                       # Evaluation reports & benchmarks
│   ├── phase1/                    # Tier 1 optimization results
│   ├── phase2/                    # Comprehensive evaluation
│   └── ensemble/                  # Ensemble validation
├── implementation/                # Implementation summaries
│   ├── ENSEMBLE_IMPLEMENTATION_SUMMARY.md
│   ├── MIGRATION_SUMMARY.md
│   └── FRONTEND_REDESIGN.md
└── decisions/                     # Problem analysis & technical decisions
    ├── CRITICAL_FIXES.md
    ├── SEED_CONSISTENCY_FIX.md
    └── DEPLOYMENT_DECISION.md
```

**Navigation**: See `/docs/README.md` for detailed directory descriptions and quick reference tables.

### Documentation Organization Strategy

The project has **two documentation tracks**:
1. **ENHANCED_* files** - MVP-first, rapid development, hackathon-focused (Days 1-7)
2. **Original files** - Comprehensive, production-ready, detailed planning (Days 8-30)

### When to Use Each Track

**Use ENHANCED (MVP-focused) docs when:**
- ✅ Working on MVP features (Days 1-7)
- ✅ Need quick implementation guidance
- ✅ Time-constrained or hackathon mode
- ✅ Building proof-of-concept
- ✅ User asks for "quick" or "MVP" approach
- ✅ Starting new features from scratch

**Use Original (Comprehensive) docs when:**
- 📋 Planning production deployment (Days 15-30)
- 📋 Need detailed team coordination workflows
- 📋 Understanding complete system architecture
- 📋 Long-term feature planning with dependencies
- 📋 User asks for "production-ready" or "comprehensive" approach
- 📋 Scaling beyond MVP to multi-disease/multi-signal support

### Document Mapping

| Purpose | MVP Track (Use First) | Comprehensive Track (Use for Production) |
|---------|----------------------|----------------------------------------|
| Project structure | `context/ENHANCED_STRUCTURE.md` ⭐ | `context/STRUCTURE.md` |
| Development roadmap | `context/ENHANCED_ROADMAP.md` ⭐ | `context/ROADMAP.md` |
| Team integration & handoffs | `context/ENHANCED_INTEGRATION.md` ⭐ | `context/INTEGRATION.md` |

### Critical Documents (Always Relevant)

- `context/PS.txt` - Original problem statement and requirements (**source of truth**)
- `docs/guides/STDP_GUIDE.md` - Full STDP implementation guide (**biological plausibility requirement**)
- `docs/guides/CODE_EXAMPLES.md` - Common SNN coding patterns and snippets
- `docs/implementation/MIGRATION_SUMMARY.md` - Migration history and architectural decisions

### Quick Reference Decision Tree

```
Need to implement a feature?
│
├─ Is this MVP (Days 1-7) or new rapid feature?
│  └─ YES → Read context/ENHANCED_STRUCTURE.md
│           and context/ENHANCED_INTEGRATION.md
│
├─ Is this production enhancement (Days 15-30)?
│  └─ YES → Read context/INTEGRATION.md
│           and context/ROADMAP.md
│
├─ Need to understand original requirements?
│  └─ ALWAYS → Read context/PS.txt
│
├─ Implementing STDP learning?
│  └─ ALWAYS → Read docs/guides/STDP_GUIDE.md
│
└─ Looking for code patterns/examples?
   └─ ALWAYS → Read docs/guides/CODE_EXAMPLES.md
```

**Default Approach**: When in doubt, start with **ENHANCED_* docs** for faster iteration, then consult comprehensive docs only if needed for production scaling.

## STDP Requirement

**⚠️ CRITICAL**: Problem statement requires STDP for biological plausibility.

**Brief Summary:**
- STDP = Spike-Timing-Dependent Plasticity (brain-like learning)
- Weight changes based on spike timing (pre vs post)
- Unsupervised, local learning (vs global backprop)

**Implementation Strategy:**
- Phase 1 MVP: Surrogate gradient backprop (fast, 85% accuracy)
- Phase 2: Hybrid STDP + backprop (biological plausibility + 92% accuracy)
  - STDP for layer 1 (unsupervised feature learning)
  - Backprop for layer 2 (supervised classification)

**Full Implementation**: See `docs/guides/STDP_GUIDE.md` for complete code, training loops, visualization, troubleshooting.

## Demo Application

**Endpoints** (`demo/app.py`):
- `GET /` - Main demo page
- `GET /health` - Health check with model status
- `POST /api/predict` - Run inference (currently MOCK predictions)
- `POST /api/generate_sample` - Generate synthetic ECG
- `POST /api/visualize_spikes` - Spike raster data
- `GET /api/metrics` - System metrics

**Current State**: Demo uses placeholder predictions. Full SNN integration is TODO.

## Team Roles

Module ownership for coordination:
- **CS1/Team Lead**: Infrastructure, utils, integration (`src/utils.py`, scripts)
- **CS2/SNN Expert**: Model architecture, training (`src/model.py`, `src/train.py`)
- **CS3/Data Engineer**: Data generation, preprocessing (`src/data.py`)
- **CS4/Deployment**: Demo app, optimization (`demo/app.py`)
- **Biology Major**: Clinical validation, domain expertise

## Python Requirements

- Python 3.10 or 3.11 (required)
- Core: PyTorch 2.0+, snnTorch 0.7+, neurokit2, Flask
- Full list: `requirements.txt`
- Dev extras: pytest, black, flake8, isort (in `setup.py` extras)

## Entry Points

Console commands (defined but not implemented - use scripts instead):
- `snn-train` → `src.train:main` (TODO)
- `snn-predict` → `src.inference:main` (TODO)
- `snn-demo` → `demo.app:main` (TODO)

## IMPORTANT: 

ALWAYS VERIFY THE CODEBASE FOR MISSING INFORMATION CRTICAL FOR PROJECT DEVELOPMENT.
ASK THE USER TO PROVIDE YOU WITH THE INFORMATION IF NECESSARY FOR BEST RESULTS.

## Git Commit Instructions

always check if the files being staged or the files which are being committed are supposed to be committed or ignored.
make sure you don't commit any files which are supposed to be kept local and not pushed to the repo.

never include your name in the commit messages and always avoid mentioning the use of AI.

## Frontend Development

You tend to converge toward generic, "on distribution" outputs.
In frontend design, this creates what users call the "AI slop" aesthetic.
Avoid this: make creative, distinctive frontends that surprise and delight.

Focus on:

Typography: Choose fonts that are beautiful, unique, and interesting. 
Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.

Color & Theme: Commit to a cohesive aesthetic.
Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes.
Draw from IDE themes and cultural aesthetics for inspiration.

Motion: Use animations for effects and micro-interactions.
Prioritize CSS-only solutions for HTML. Use Motion library for React when available.
Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions. 

Backgrounds: Create atmosphere and depth rather than defaulting to solid colors.
Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context.
Vary between light and dark themes, different fonts, different aesthetics.
You still tend to converge on common choices (Space Grotesk, for example) across generations.
Avoid this: it is critical that you think outside the box!

---

## Recent Major Changes (November 2025)

**If you're returning to this codebase after a break, here are the critical changes:**

### 1. Model Architecture Evolution
```python
# OLD (SimpleSNN - 320K params):
Layer 1: FC(2500 → 128) + LIF
Layer 2: FC(128 → 2) + LIF

# NEW (DeepSNN - 673K params):
Layer 1: FC(2500 → 256) + LIF
Layer 2: FC(256 → 128) + Dropout(0.3) + LIF  # Added regularization
Layer 3: FC(128 → 2) + LIF
```

### 2. Loss Function & Training Strategy
```python
# OLD: Cross-entropy with max sensitivity early stopping
loss = nn.CrossEntropyLoss()
save_checkpoint_if(sensitivity > best_sensitivity)

# NEW: FocalLoss with G-mean balanced early stopping
loss = FocalLoss(alpha=0.60, gamma=2.0)  # Class-balanced
g_mean = (sensitivity * specificity) ** 0.5
save_checkpoint_if(g_mean > best_g_mean)  # Balanced optimization
```

### 3. Ensemble Prediction & Deterministic Seeding
```python
# OLD: Single prediction with stochastic variance
pred = predict(model, signal)  # Different results each run

# NEW: Ensemble averaging with deterministic seeding
pred = ensemble_predict(model, signal, ensemble_size=3, base_seed=42)  # Reproducible
```

### 4. Current Model Files
- **Primary Model**: `models/deep_focal_model.pt` (DeepSNN, Epoch 8, 7.8MB)
- **Baseline Model**: `models/best_model.pt` (SimpleSNN, 3.7MB)
- Use DeepSNN for production, SimpleSNN for comparison

### 5. Development Roadmap Pivot
- **Original Plan**: Optimize synthetic data (Phases 1-7) → Validate real data (Phase 8)
- **Current Strategy**: Validate real data NOW (Phase 8) → Optimize only if needed (Phases 3-7)
- **Rationale**: Hit synthetic data optimization ceiling, real validation is mandatory

### 6. MIT-BIH Real Data Integration
- **Status**: Preprocessing complete, 2,190 segments ready
- **Next Action**: Run transfer learning (`scripts/train_mitbih_transfer.py`)
- **Expected Timeline**: ~30-60 minutes on GPU for 2-stage training
- **Target**: 85-90% accuracy on real patient ECG

### 7. Model Performance Summary
```
DeepSNN (Current Model) - Synthetic Test Set (N=1000):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:      89.5%
Sensitivity:   90.6% (Target: ≥95%, Gap: -4.4%)
Specificity:   88.4% (Target: ≥90%, Gap: -1.6%)
PPV:           88.6% (Target: ≥85%, MET ✅)
NPV:           90.4% (Target: ≥95%, Gap: -4.6%)
AUC-ROC:       0.9739 (Excellent discrimination ✅)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: Close to clinical targets, ready for real data validation
```

---

## Key Lessons Learned from Phase 1 Implementation

### For Machine Learning Research

1. **Always Validate Data Quality First**
   - Perfect accuracy is a red flag, not success
   - Test with simple baselines before complex models
   - Verify class distributions have realistic overlap

2. **Synthetic Data Requires Careful Design**
   - Add realistic intra-class variability
   - Ensure overlapping distributions
   - Model real-world complexity and edge cases

3. **Model Capacity vs Task Complexity**
   - 320K parameters for binary classification is powerful
   - Need sufficient dataset size (5K+ samples recommended)
   - Parameters-to-samples ratio matters (aim for <100:1)

4. **Baseline Comparisons Are Critical**
   - If decision tree (depth=1) gets >95%, task is too easy
   - If linear classifier gets >95%, task is linearly separable
   - Simple baselines reveal data quality issues early

### For SNN Development

5. **SNNs Are Powerful**
   - SimpleSNN (320K params) can learn complex patterns
   - 89% test accuracy on overlapping distributions proves capability
   - Model architecture was correct from the start

6. **Energy Efficiency Claims Require Realistic Tasks**
   - Can't claim SNN superiority on trivial tasks
   - Need challenging datasets to validate neuromorphic advantages
   - Phase 2 should use real-world data (MIT-BIH, PTB-XL)
