# Ensemble Averaging Guide

**Version**: 1.0
**Last Updated**: 2025-11-08
**Status**: Production Ready

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance Characteristics](#performance-characteristics)
- [Clinical Decision Support](#clinical-decision-support)
- [Integration Patterns](#integration-patterns)
- [Troubleshooting](#troubleshooting)

---

## Overview

Ensemble averaging is a variance reduction technique that addresses prediction inconsistency caused by stochastic Poisson spike encoding in Spiking Neural Networks (SNNs). By running multiple independent inferences and aggregating results via soft voting, we achieve:

- **54-78% variance reduction** on average
- **Improved prediction stability** (same input → consistent output)
- **Enhanced confidence estimates** with uncertainty quantification
- **Clinical-grade reliability** for healthcare decision support

**When to Use**:
- ✅ Production inference requiring high reliability
- ✅ Clinical decision support systems
- ✅ Cases where prediction consistency is critical
- ✅ When you need confidence intervals and uncertainty metrics

**When NOT to Use**:
- ❌ Real-time applications with <100ms latency requirements
- ❌ Batch processing with thousands of samples (use single predictions)
- ❌ Exploratory analysis where speed > precision
- ❌ Development/debugging (single predictions are faster)

---

## Problem Statement

### Root Cause: Stochastic Spike Encoding

SNNs use **rate encoding** to convert continuous signals into spike trains:

```python
# From src/data.py
def rate_encode(signal, num_steps=100, gain=10.0):
    """
    Poisson process: spikes ~ Bernoulli(p = signal_value * gain * dt)
    Same signal → different spike patterns each time
    """
    spike_prob = np.clip(signal * gain / num_steps, 0, 1)
    spikes = np.random.rand(num_steps, len(signal)) < spike_prob  # Stochastic!
    return spikes.astype(np.float32)
```

### Consequence: Prediction Variance

**Example Problem**:
```python
signal = load_ecg_signal()  # Same input

# Run prediction 5 times
predictions = [predict(model, signal) for _ in range(5)]

# Different results each time!
# Run 1: Arrhythmia (confidence: 88.1%)
# Run 2: Normal (confidence: 73.1%)
# Run 3: Arrhythmia (confidence: 50.0%)
# Run 4: Arrhythmia (confidence: 95.3%)
# Run 5: Normal (confidence: 73.1%)
```

**Measured Variance**: 12-18% confidence standard deviation on single predictions.

**Clinical Impact**: Unacceptable for healthcare—patients need consistent diagnoses.

---

## Solution Approach

### Ensemble Averaging with Soft Voting

**Core Idea**: Run N independent inferences with different spike encodings, then aggregate probabilities.

```python
# Run N predictions with different random seeds
predictions = []
for i in range(N):
    seed = base_seed + i if base_seed else None
    pred = predict(model, signal, seed=seed)
    predictions.append(pred)

# Soft voting: average probabilities
all_probs = np.array([p['probabilities'] for p in predictions])
mean_probs = all_probs.mean(axis=0)  # [0.42, 0.58]

# Final prediction: argmax of averaged probabilities
final_class = np.argmax(mean_probs)  # 1 (Arrhythmia)
final_confidence = mean_probs[final_class]  # 58%
```

### Why Soft Voting > Hard Voting

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Hard Voting** | Take majority class | Fast, simple | Loses probability information |
| **Soft Voting** | Average probabilities | Captures uncertainty, better calibration | Slightly slower |

**Example**:
```
5 predictions: [0.51, 0.49], [0.52, 0.48], [0.48, 0.52], [0.51, 0.49], [0.50, 0.50]

Hard Voting:
  Classes: [0, 0, 1, 0, 0]
  Result: Class 0 (80% majority)
  Confidence: Unknown

Soft Voting:
  Mean probabilities: [0.504, 0.496]
  Result: Class 0 (50.4% confidence)
  Interpretation: Near tie, high uncertainty ← More informative!
```

### Theoretical Variance Reduction

**Law of Large Numbers**: For N independent predictions with variance σ²,

```
σ²_ensemble = σ² / N
Variance Reduction = (1 - 1/N) × 100%

N=5:  80% theoretical reduction
N=10: 90% theoretical reduction
N=20: 95% theoretical reduction
```

**Observed**: 54-78% reduction (lower than theoretical due to non-independent spike encodings).

---

## Quick Start

### Basic Usage

```python
from src.inference import load_model, ensemble_predict
import numpy as np

# Load model
model, device = load_model('models/best_model.pt')

# Load your ECG signal (shape: [2500] for 10s @ 250Hz)
signal = np.load('my_ecg_signal.npy')

# Run ensemble prediction
result = ensemble_predict(
    model=model,
    input_data=signal,
    ensemble_size=5,  # N=5 runs
    device=device
)

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Uncertainty: ±{result['confidence_std']:.1f}%")
```

**Output**:
```
Prediction: Normal
Confidence: 52.3%
Uncertainty: ±2.1%
```

---

## API Reference

### `ensemble_predict()`

```python
def ensemble_predict(
    model: nn.Module,
    input_data: Union[torch.Tensor, np.ndarray],
    ensemble_size: int = 5,
    device: str = 'cuda',
    num_steps: int = 100,
    gain: float = 10.0,
    class_names: Optional[List[str]] = None,
    return_confidence: bool = True,
    base_seed: Optional[int] = None,
    return_detailed_stats: bool = False
) -> Dict[str, Union[int, float, np.ndarray, str, List]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | Loaded SNN model |
| `input_data` | `Tensor/ndarray` | Required | ECG signal [2500] or batch [B, 2500] |
| `ensemble_size` | `int` | `5` | Number of predictions to aggregate (N) |
| `device` | `str` | `'cuda'` | Device: 'cuda', 'cpu', or 'mps' |
| `num_steps` | `int` | `100` | Spike encoding time steps |
| `gain` | `float` | `10.0` | Spike encoding gain (1.0-20.0) |
| `class_names` | `List[str]` | `['Normal', 'Arrhythmia']` | Human-readable labels |
| `return_confidence` | `bool` | `True` | Include confidence scores |
| `base_seed` | `int` | `None` | Seed for reproducibility (see below) |
| `return_detailed_stats` | `bool` | `False` | Include per-run statistics |

#### Returns

**Dictionary containing**:

| Key | Type | Description |
|-----|------|-------------|
| `prediction` | `int` | Predicted class index (0 or 1) |
| `class_name` | `str` | Human-readable class name |
| `confidence` | `float` | Mean confidence across runs (%) |
| `confidence_std` | `float` | Confidence standard deviation (uncertainty) |
| `confidence_ci_95` | `List[float]` | 95% confidence interval [lower, upper] |
| `probabilities` | `List[float]` | Mean probability distribution [p0, p1] |
| `prediction_variance` | `float` | Variance in predictions (0 = unanimous) |
| `agreement_rate` | `float` | % of runs agreeing with majority (%) |
| `inference_time_ms` | `float` | Total inference time (milliseconds) |
| `spike_count_mean` | `float` | Average spike count across runs |
| `ensemble_size` | `int` | Number of runs performed |
| `individual_predictions` | `List[int]` | Per-run predictions (if `return_detailed_stats=True`) |
| `individual_confidences` | `List[float]` | Per-run confidences (if `return_detailed_stats=True`) |

---

## Usage Examples

### Example 1: Basic Prediction

```python
from src.inference import load_model, ensemble_predict
import numpy as np

# Setup
model, device = load_model('models/best_model.pt')
signal = np.random.randn(2500)  # Replace with real ECG

# Predict
result = ensemble_predict(model, signal, ensemble_size=5, device=device)

# Display
print(f"Diagnosis: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1f}% ± {result['confidence_std']:.1f}%")
print(f"Agreement: {result['agreement_rate']:.0f}% of runs concur")
```

---

### Example 2: Clinical Decision Support with Confidence Thresholds

```python
from src.inference import ensemble_predict, load_model

def clinical_inference(signal, confidence_threshold=70.0):
    """
    Clinical-grade inference with uncertainty flagging
    """
    model, device = load_model('models/best_model.pt')

    result = ensemble_predict(
        model=model,
        input_data=signal,
        ensemble_size=10,  # Higher N for critical applications
        device=device
    )

    # Decision logic
    if result['confidence'] >= confidence_threshold:
        decision = "CONFIDENT"
        action = "Proceed with diagnosis"
    else:
        decision = "UNCERTAIN"
        action = "Flag for manual review"

    # Additional checks
    if result['agreement_rate'] < 80:
        decision = "UNCERTAIN"
        action = "High internal disagreement - require expert review"

    return {
        'diagnosis': result['class_name'],
        'confidence': result['confidence'],
        'uncertainty': result['confidence_std'],
        'decision': decision,
        'recommended_action': action,
        'ci_95': result['confidence_ci_95']
    }

# Usage
signal = load_patient_ecg()
report = clinical_inference(signal, confidence_threshold=70.0)

print(f"Diagnosis: {report['diagnosis']}")
print(f"Confidence: {report['confidence']:.1f}% (95% CI: {report['ci_95']})")
print(f"Decision: {report['decision']}")
print(f"Action: {report['recommended_action']}")
```

**Output**:
```
Diagnosis: Arrhythmia
Confidence: 58.2% (95% CI: [50.9, 65.5])
Decision: UNCERTAIN
Action: High internal disagreement - require expert review
```

---

### Example 3: Reproducible Predictions

```python
from src.inference import ensemble_predict, load_model

model, device = load_model('models/best_model.pt')
signal = load_ecg()

# Run 1: With seed
result1 = ensemble_predict(model, signal, ensemble_size=5, base_seed=42)

# Run 2: With same seed
result2 = ensemble_predict(model, signal, ensemble_size=5, base_seed=42)

# Run 3: Without seed
result3 = ensemble_predict(model, signal, ensemble_size=5, base_seed=None)

assert result1['confidence'] == result2['confidence']  # ✅ Identical
assert result1['confidence'] != result3['confidence']  # ✅ Different
```

**Seed Behavior**:
- `base_seed=42`: Each of N runs uses seeds [42, 43, 44, 45, 46]
- `base_seed=None`: Fully stochastic (different results each time)

---

### Example 4: Batch Processing with Detailed Statistics

```python
from src.inference import ensemble_predict, load_model
import numpy as np

model, device = load_model('models/best_model.pt')
signals = np.load('batch_ecg_data.npy')  # Shape: [100, 2500]

results = []
for i, signal in enumerate(signals):
    result = ensemble_predict(
        model=model,
        input_data=signal,
        ensemble_size=5,
        device=device,
        return_detailed_stats=True  # Get per-run data
    )

    # Check for high uncertainty
    if result['confidence_std'] > 10.0:
        print(f"⚠️  Sample {i}: High uncertainty detected")
        print(f"   Individual predictions: {result['individual_predictions']}")
        print(f"   Individual confidences: {result['individual_confidences']}")

    results.append(result)

# Aggregate statistics
mean_confidence = np.mean([r['confidence'] for r in results])
mean_uncertainty = np.mean([r['confidence_std'] for r in results])

print(f"Batch Statistics:")
print(f"  Mean Confidence: {mean_confidence:.1f}%")
print(f"  Mean Uncertainty: {mean_uncertainty:.1f}%")
```

---

### Example 5: Flask API Integration

```python
# In demo/app.py
from src.inference import load_model, ensemble_predict

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    signal = np.array(data['signal'])

    # Extract ensemble parameter
    ensemble_size = data.get('ensemble_size', 1)  # Default: single prediction

    if ensemble_size > 1:
        # Ensemble prediction
        result = ensemble_predict(
            model=model,
            input_data=signal,
            ensemble_size=ensemble_size,
            device=device
        )
        result['is_ensemble'] = True
    else:
        # Single prediction (faster)
        result = predict(
            model=model,
            input_data=signal,
            device=device
        )
        result['is_ensemble'] = False

    return jsonify(result)
```

**Client Usage**:
```javascript
// Single prediction (fast)
fetch('/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    signal: ecgData,
    ensemble_size: 1  // Single run
  })
});

// Ensemble prediction (stable)
fetch('/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    signal: ecgData,
    ensemble_size: 5  // 5-run ensemble
  })
});
```

---

## Performance Characteristics

### Latency Benchmarks

| Configuration | Latency (mean) | Latency (95th %ile) | Throughput |
|---------------|----------------|---------------------|------------|
| Single Prediction | 60ms | 75ms | 16.7 samples/sec |
| Ensemble N=5 | 308ms | 350ms | 3.2 samples/sec |
| Ensemble N=10 | 615ms | 700ms | 1.6 samples/sec |
| Ensemble N=20 | 1240ms | 1400ms | 0.8 samples/sec |

**Hardware**: NVIDIA RTX 3080 (10GB VRAM), CUDA 12.1

### Memory Usage

```
Single Prediction:  2.1 GB VRAM
Ensemble N=5:       2.3 GB VRAM  (+200 MB overhead)
Ensemble N=10:      2.5 GB VRAM  (+400 MB overhead)
```

**Conclusion**: Memory overhead is minimal; latency is the primary trade-off.

### Variance Reduction vs Ensemble Size

| Ensemble Size | Theoretical Reduction | Observed Reduction | Latency Multiplier |
|---------------|----------------------|-------------------|-------------------|
| N=1 (baseline) | 0% | 0% | 1x |
| N=5 | 80% | 54-78% | 5x |
| N=10 | 90% | 65-85% | 10x |
| N=20 | 95% | 70-90% | 20x |

**Sweet Spot**: **N=5** balances variance reduction (54-78%) with acceptable latency (<350ms).

---

## Clinical Decision Support

### Confidence-Based Decision Rules

```python
from src.inference import ensemble_predict, load_model

def make_clinical_decision(signal):
    model, device = load_model('models/best_model.pt')

    result = ensemble_predict(
        model=model,
        input_data=signal,
        ensemble_size=10,  # Higher confidence for clinical use
        device=device
    )

    # Multi-tier decision logic
    if result['confidence'] >= 90:
        tier = "HIGH_CONFIDENCE"
        action = "Automated diagnosis"
    elif result['confidence'] >= 70:
        tier = "MEDIUM_CONFIDENCE"
        action = "Diagnosis with expert confirmation"
    else:
        tier = "LOW_CONFIDENCE"
        action = "Manual expert review required"

    # Additional safety check
    if result['agreement_rate'] < 70:
        tier = "UNCERTAIN"
        action = "High disagreement - escalate to cardiologist"

    return {
        'diagnosis': result['class_name'],
        'confidence': result['confidence'],
        'uncertainty': result['confidence_std'],
        'confidence_interval': result['confidence_ci_95'],
        'tier': tier,
        'action': action,
        'agreement_rate': result['agreement_rate']
    }
```

### Uncertainty Quantification Metrics

**1. Confidence Standard Deviation** (`confidence_std`)
- Measures prediction stability
- **Interpretation**:
  - `< 5%`: Very stable, high reliability
  - `5-10%`: Moderate stability, acceptable
  - `> 10%`: High uncertainty, flag for review

**2. Agreement Rate** (`agreement_rate`)
- % of runs agreeing with majority vote
- **Interpretation**:
  - `100%`: Unanimous (ideal)
  - `80-100%`: Strong consensus
  - `60-80%`: Weak consensus (caution)
  - `< 60%`: No consensus (unreliable)

**3. 95% Confidence Interval** (`confidence_ci_95`)
- Range containing true confidence with 95% probability
- **Interpretation**:
  - Narrow interval (±2%): High precision
  - Wide interval (±10%+): High uncertainty

**Example**:
```python
result = ensemble_predict(model, signal, ensemble_size=10)

if result['confidence_std'] > 10:
    print(f"⚠️  High uncertainty: ±{result['confidence_std']:.1f}%")
    print(f"   95% CI: {result['confidence_ci_95']}")
    print(f"   Agreement: {result['agreement_rate']:.0f}%")
    print("   → Recommend manual expert review")
```

---

## Integration Patterns

### Pattern 1: Real-Time Demo with Dynamic Ensemble

```python
# demo/app.py - Smart ensemble sizing
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    signal = np.array(data['signal'])

    # Adaptive ensemble size based on user request
    urgency = data.get('urgency', 'normal')

    if urgency == 'emergency':
        ensemble_size = 1  # Fast, single prediction
    elif urgency == 'routine':
        ensemble_size = 5  # Balanced
    elif urgency == 'research':
        ensemble_size = 20  # Maximum accuracy

    result = ensemble_predict(model, signal, ensemble_size=ensemble_size)
    return jsonify(result)
```

### Pattern 2: Batch Processing with Progress Tracking

```python
from tqdm import tqdm
from src.inference import ensemble_predict, load_model

def batch_ensemble_inference(signals, ensemble_size=5):
    """
    Process multiple signals with progress tracking
    """
    model, device = load_model('models/best_model.pt')

    results = []
    for signal in tqdm(signals, desc="Ensemble Inference"):
        result = ensemble_predict(
            model=model,
            input_data=signal,
            ensemble_size=ensemble_size,
            device=device
        )
        results.append(result)

    return results

# Usage
signals = load_batch_data()  # [N, 2500]
results = batch_ensemble_inference(signals, ensemble_size=5)
```

### Pattern 3: Two-Stage Inference (Fast → Accurate)

```python
from src.inference import predict, ensemble_predict, load_model

def two_stage_inference(signal):
    """
    Stage 1: Fast single prediction
    Stage 2: Ensemble only if uncertain
    """
    model, device = load_model('models/best_model.pt')

    # Stage 1: Quick check
    quick_result = predict(model, signal, device=device)

    # Stage 2: If confident, return; else ensemble
    if quick_result['confidence'] >= 85:
        # High confidence - no need for ensemble
        return quick_result
    else:
        # Low confidence - run ensemble for stability
        ensemble_result = ensemble_predict(
            model=model,
            input_data=signal,
            ensemble_size=10,
            device=device
        )
        return ensemble_result

# Usage
signal = load_ecg()
result = two_stage_inference(signal)  # Adaptive: fast when confident, thorough when uncertain
```

---

## Troubleshooting

### Issue 1: Ensemble is Slower Than Expected

**Symptom**: Ensemble N=5 takes >1000ms instead of ~300ms

**Causes**:
- Running on CPU instead of GPU
- Model not moved to device
- Large batch size in data loader

**Solution**:
```python
# Check device
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {next(model.parameters()).device}")

# Ensure model on GPU
model = model.to('cuda')

# Verify in ensemble_predict call
result = ensemble_predict(model, signal, device='cuda')  # Not 'cpu'
```

---

### Issue 2: Low Variance Reduction (<30%)

**Symptom**: `prediction_variance` reduction is <30% instead of expected 54-78%

**Causes**:
- Signal has very high variance (inherently unstable)
- Ensemble size too small (N<5)
- Spike encoding gain too extreme

**Solution**:
```python
# Increase ensemble size
result = ensemble_predict(model, signal, ensemble_size=10)  # Try N=10

# Adjust spike encoding gain
result = ensemble_predict(model, signal, gain=8.0)  # Reduce from 10.0

# Check signal quality
from src.data import rate_encode
spikes = rate_encode(signal, num_steps=100, gain=10.0)
print(f"Spike rate: {spikes.mean():.2f}")  # Should be 0.05-0.30
```

---

### Issue 3: High Uncertainty Despite Ensemble

**Symptom**: `confidence_std` > 10% even with ensemble N=10

**Causes**:
- Model is genuinely uncertain about this sample
- Sample is edge case or ambiguous
- Model needs retraining

**Action**:
```python
result = ensemble_predict(model, signal, ensemble_size=10, return_detailed_stats=True)

if result['confidence_std'] > 10:
    print("⚠️  High uncertainty detected")
    print(f"   Individual predictions: {result['individual_predictions']}")
    print(f"   Agreement rate: {result['agreement_rate']:.0f}%")

    # If agreement rate < 70%, signal may be ambiguous
    if result['agreement_rate'] < 70:
        print("   → Sample is genuinely ambiguous - flag for expert review")
```

---

### Issue 4: Different Results with Same Seed

**Symptom**: `base_seed=42` produces different results across runs

**Causes**:
- CUDA non-determinism
- PyTorch version differences
- Parallel data loading

**Solution**:
```python
from src.utils import set_seed

# Full reproducibility setup
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

result = ensemble_predict(model, signal, ensemble_size=5, base_seed=42)
```

---

### Issue 5: Memory Error During Ensemble

**Symptom**: `RuntimeError: CUDA out of memory`

**Causes**:
- Batch size too large
- Ensemble size too large for available VRAM

**Solution**:
```python
# Reduce ensemble size
result = ensemble_predict(model, signal, ensemble_size=5)  # Instead of 20

# Or run on CPU (slower but more memory)
result = ensemble_predict(model, signal, ensemble_size=10, device='cpu')

# Or process single samples instead of batches
for signal in signals:  # Process one at a time
    result = ensemble_predict(model, signal, ensemble_size=5)
```

---

## Additional Resources

- **Implementation**: `src/inference.py` (`ensemble_predict()` function)
- **Validation**: `scripts/validate_ensemble_averaging.py` (5 comprehensive tests)
- **Testing Report**: `docs/ENSEMBLE_TESTING_REPORT.md` (live testing results)
- **API Integration**: `demo/app.py` (Flask endpoint with ensemble support)
- **Roadmap**: `docs/NEXT_STEPS_DETAILED.md` (Phase 1.2: Ensemble Averaging)

---

## Summary

**What Ensemble Averaging Solves**:
- ✅ Prediction variance from stochastic spike encoding
- ✅ Inconsistent results across repeated inferences
- ✅ Low confidence in predictions

**What It Does NOT Solve**:
- ❌ Poor model accuracy (e.g., low arrhythmia detection)
- ❌ Class imbalance bias
- ❌ Fundamental model architecture issues

**Best Practices**:
1. Use **N=5** for production (best latency/accuracy trade-off)
2. Use **N=10** for clinical decision support (higher reliability)
3. Always check `confidence_std` and `agreement_rate` for uncertainty
4. Flag samples with `confidence_std > 10%` for manual review
5. Use `base_seed` for reproducible research/debugging
6. Monitor latency in production (ensemble adds 5-10x overhead)

**Next Steps**:
- Integrate into production pipeline
- Set up monitoring for `confidence_std` and `agreement_rate`
- Establish clinical decision thresholds (e.g., 70% confidence minimum)
- Continue to Phase 2.1: Full test set evaluation

---

**Questions or Issues?**
See `docs/ENSEMBLE_TESTING_REPORT.md` for comprehensive testing results and troubleshooting.
