# CLAUDE.md

Concise project reference for Claude Code. For detailed guides, see `docs/` directory.

## Project Overview

**Neuromorphic SNN for Healthcare Signal Pattern Recognition** (ECG/EEG analysis)

**Goals:**
- 60%+ energy efficiency vs CNNs
- 92%+ accuracy on disease detection
- <50ms inference time
- **CRITICAL**: STDP implementation required for biological plausibility (see `docs/STDP_GUIDE.md`)

**Current Status:**
- ✅ SNN simulation on GPU (snnTorch)
- ✅ Synthetic ECG data generation
- ✅ Surrogate gradient backprop (working baseline)
- ⚠️ STDP: Phase 1 uses backprop, Phase 2 will add hybrid STDP

**Success Metrics:**
- Minimum Viable: 85% accuracy, 40% energy efficiency, <100ms inference
- Target: 92% accuracy, 60% energy efficiency, <50ms inference

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

### Module Map

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/model.py` | SNN architectures | `SimpleSNN`, `measure_energy_efficiency()` |
| `src/data.py` | Data generation & encoding | `generate_synthetic_ecg()`, `rate_encode()`, `ECGDataset` |
| `src/train.py` | Training pipeline | `train_epoch()`, `validate()`, `train_model()` |
| `src/inference.py` | Prediction & loading | `load_model()`, `predict()` |
| `src/utils.py` | Utilities | `set_seed()`, `get_device()` |
| `demo/app.py` | Flask web demo | Endpoints: `/`, `/health`, `/api/predict` |

### Data Flow

```
src/data.py → src/model.py → src/train.py → src/inference.py → demo/app.py
    ↓              ↓             ↓              ↓                ↓
Generate       Define SNN    Train with     Run Predict      Web Demo
Synthetic      LIF Neurons   Surrogate      On Signals       User UI
ECG/EEG                      Gradients
```

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

### snnTorch Gotchas

**State Device Mismatch**
```python
model.to(device)
mem = lif.init_leaky().to(device)  # State must match model device!
```

**No Time Parallelization**
```python
# Must iterate sequentially over time dimension
for step in range(x.size(0)):  # Cannot parallelize
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
# inference.py:load_model() handles both checkpoint dict and raw state_dict
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

**SimpleSNN Architecture** (`src/model.py`):
- Input: 2500 (10s × 250Hz)
- Hidden: 128
- Output: 2 classes
- Beta (membrane decay): 0.9

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

## Common Issues

**CUDA Out of Memory**
- Reduce `BATCH_SIZE` in `.env` (try 16 for 4GB VRAM, 8 if still failing)
- SNNs process sequential time steps → higher memory than ANNs

**Model Not Converging**
- Adjust `LEARNING_RATE` (try 0.01 or 0.0001)
- Check spike encoding gain (default 10.0, range 1.0-20.0)
- Verify surrogate gradient function (fast_sigmoid recommended)

**Spike Encoding Issues**
- All 1s: `gain` too high → reduce to 5.0-10.0
- All 0s: `gain` too low → increase to 10.0-15.0
- Verify input signal normalized to [0, 1] range

**Non-Reproducible Results**
- `rate_encode()` is stochastic (Poisson process)
- Call `set_seed(42)` before encoding

**SNN State Initialization Error**
- Forgot to initialize: `mem = lif.init_leaky()`
- Must call before forward pass

**Device Mismatch**
- Model on GPU, state on CPU (or vice versa)
- Solution: `mem = lif.init_leaky().to(device)`

**Training Script Uses Baseline ANN**
- `scripts/03_train_mvp_model.sh` uses ANN by default (intentional for MVP)
- To use SimpleSNN: modify script to import from `src.model`

**Demo Returns Mock Predictions**
- Expected for MVP - SNN integration is TODO in `demo/app.py`

**Import Errors After Installation**
- Verify venv activated: `which python` should show venv path
- Reinstall: `pip install -r requirements.txt --force-reinstall`
- For CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

**STDP Issues**
- See `docs/STDP_GUIDE.md` for detailed troubleshooting
- Common: weights saturate, learning too slow, no spike coincidences, memory issues

## Context & Documentation

### Context Directory (`context/`)

- `PS.txt` - Original problem statement, requirements
- `STRUCTURE.md` - Complete ideal project structure
- `ENHANCED_STRUCTURE.md` - **MVP-focused simplified structure** (recommended)
- `ROADMAP.md` - Detailed 30-day development roadmap
- `ENHANCED_ROADMAP.md` - **AI-assisted rapid development** (recommended)
- `INTEGRATION.md` - Module integration timeline
- `ENHANCED_INTEGRATION.md` - **Simplified MVP integration** (recommended)

### Documentation (`docs/`)

- `STDP_GUIDE.md` - **Full STDP implementation guide** (biological plausibility requirement)
- `CODE_EXAMPLES.md` - Common coding patterns and snippets

**When to refer:**
- Starting features → `ENHANCED_STRUCTURE.md`
- Planning sprints → `ENHANCED_ROADMAP.md`
- Understanding requirements → `PS.txt`
- STDP implementation → `docs/STDP_GUIDE.md`
- Code patterns → `docs/CODE_EXAMPLES.md`

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

**Full Implementation**: See `docs/STDP_GUIDE.md` for complete code, training loops, visualization, troubleshooting.

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
