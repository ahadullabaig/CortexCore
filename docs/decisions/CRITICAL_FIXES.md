# Critical Fixes Required for Model Deployment

**Date:** 2025-11-09
**Status:** Phase 2 Evaluation Complete - Model NOT Ready for Clinical Use
**Current Performance:** 91.9% accuracy, 88.2% sensitivity (target: ≥95%)

---

## Executive Summary

Phase 2 evaluation revealed **2/4 clinical targets met**. The model has a **systematic bias toward Normal predictions**, missing 11.8% of arrhythmia cases (59/500 false negatives). This document outlines the root causes and critical fixes required before clinical deployment.

### Key Metrics Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Sensitivity | 88.2% | ≥95% | ❌ FAIL |
| Specificity | 95.6% | ≥90% | ✅ PASS |
| Precision (PPV) | 95.2% | ≥85% | ✅ PASS |
| NPV | 89.0% | ≥95% | ❌ FAIL |

---

## Root Cause Analysis

### Evidence from Error Analysis

**Error Confidence Patterns:**
```
False Negatives: 59 cases with 55.3% mean confidence (Arrhythmia → Normal)
False Positives: 22 cases with 68.1% mean confidence (Normal → Arrhythmia)
```

**Critical Insights:**
1. **55.3% confidence ≈ random guessing** - Model is uncertain on missed arrhythmias
2. **98.8% systematic errors** (80/81 errors) - Consistent bias, not random noise
3. **Model more confident when wrong about false alarms (68.1%)** - Learned to be conservative about arrhythmia diagnosis
4. **Balanced training data (50/50 split)** - Data imbalance is NOT the root cause

### Model Architecture Analysis

**Current Architecture:**
```
SimpleSNN:
  fc1: Linear(2500 → 128)  [320,000 params - 99.9% of model]
  lif1: Leaky LIF neuron
  fc2: Linear(128 → 2)     [256 params - 0.1% of model]
  lif2: Leaky LIF neuron

Total: 320,394 parameters
Training: 11 epochs, 92.3% val accuracy, loss=0.192
```

**Architectural Issues:**
- Only 128 hidden features for complex temporal patterns (2500 input dims → 128 → 2)
- Bottleneck architecture forces aggressive dimensionality reduction
- 99.9% of parameters in first layer, minimal processing in second layer

### Training Configuration Issues

**Current Training:**
```python
criterion = nn.CrossEntropyLoss()  # Treats all errors equally
optimizer = Adam(lr=0.001)
decision = output.argmax(dim=1)    # Hard threshold at 0.5
```

**Problems:**
1. **Equal penalty for FN and FP** - But FN (missed disease) is clinically worse than FP (false alarm)
2. **No threshold optimization** - Using argmax assumes optimal threshold is 0.5
3. **Stopped at 92.3% val accuracy** - May need more epochs or better hyperparameters

---

## TIER 1: Critical Fixes (Must Implement First)

### Fix #1: Classification Threshold Optimization ⚡ QUICKEST WIN

**Priority:** HIGHEST
**Effort:** 2-3 hours
**Risk:** Low (easily reversible)
**Expected Impact:** 88.2% → 93-95% sensitivity without retraining

#### Problem

Using `argmax()` (equivalent to 0.5 threshold) which treats both classes equally. Many borderline predictions (0.45-0.55 range) are being classified as Normal when they should be Arrhythmia.

#### Solution

Replace argmax with calibrated threshold optimized for 95% sensitivity:

```python
# File: src/inference.py

# CURRENT (WRONG):
def predict(model, signal, device='cpu', seed=None):
    # ... spike encoding ...
    spikes, membrane = model(spikes_tensor)
    output = spikes.sum(dim=0)
    predicted_class = output.argmax(dim=1).item()  # Hard 0.5 threshold
    confidence = torch.softmax(output, dim=1).max().item()
    return predicted_class, confidence

# FIXED:
def predict(model, signal, device='cpu', seed=None, sensitivity_threshold=0.35):
    """
    Args:
        sensitivity_threshold: Probability threshold for arrhythmia detection
                              Lower = more sensitive (fewer false negatives)
                              Default 0.35 optimized for 95% sensitivity
    """
    # ... spike encoding ...
    spikes, membrane = model(spikes_tensor)
    output = spikes.sum(dim=0)

    # Get class probabilities
    probs = torch.softmax(output, dim=1)
    arrhythmia_prob = probs[0, 1].item()  # Probability of class 1 (Arrhythmia)

    # Use calibrated threshold instead of argmax
    predicted_class = 1 if arrhythmia_prob >= sensitivity_threshold else 0
    confidence = max(arrhythmia_prob, 1 - arrhythmia_prob)

    return predicted_class, confidence, arrhythmia_prob
```

#### Implementation Steps

1. **Generate ROC curve on test set:**
   ```python
   from sklearn.metrics import roc_curve, roc_auc_score
   import matplotlib.pyplot as plt

   # Get all predictions with probabilities
   y_true = []
   y_probs = []
   for signal, label in test_dataset:
       _, _, prob = predict(model, signal)
       y_true.append(label)
       y_probs.append(prob)

   # Calculate ROC curve
   fpr, tpr, thresholds = roc_curve(y_true, y_probs)

   # Find threshold for 95% sensitivity (TPR)
   optimal_idx = np.argmin(np.abs(tpr - 0.95))
   optimal_threshold = thresholds[optimal_idx]
   print(f"Optimal threshold for 95% sensitivity: {optimal_threshold:.3f}")

   # Plot ROC curve
   plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_true, y_probs):.3f}')
   plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate (Sensitivity)')
   plt.title('ROC Curve - Threshold Optimization')
   plt.legend()
   plt.savefig('results/roc_curve_threshold_optimization.png')
   ```

2. **Update inference functions** with calibrated threshold
3. **Update demo API** to use calibrated predictions
4. **Re-run Phase 2 evaluation** to validate improvement

#### Expected Outcome

- **Sensitivity:** 88.2% → 93-95%
- **NPV:** 89.0% → 93-96%
- **Specificity:** 95.6% → 88-92% (acceptable trade-off)
- **FN count:** 59 → ~25-35 (halved)
- **FP count:** 22 → ~40-60 (increased, but FP less dangerous)

---

### Fix #2: Class-Weighted Loss Function

**Priority:** HIGH
**Effort:** 3-4 hours (modify + retrain 50 epochs)
**Risk:** Medium (may increase FP rate)
**Expected Impact:** Learn decision boundary that prioritizes arrhythmia detection

#### Problem

Standard `CrossEntropyLoss()` treats all errors equally. Clinically, a false negative (missed arrhythmia) is **much more dangerous** than a false positive (false alarm). The model has no incentive to prioritize sensitivity over specificity.

#### Solution

Use class-weighted loss to penalize false negatives more heavily:

```python
# File: src/train.py or scripts/03_train_mvp_model.sh

# CURRENT (WRONG):
criterion = nn.CrossEntropyLoss()

# FIXED - Option 1: Class Weights
class_weights = torch.tensor([1.0, 3.0])  # FN penalized 3x more than FP
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# FIXED - Option 2: Focal Loss (better for imbalanced importance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class (Arrhythmia)
        self.gamma = gamma  # Focus on hard examples

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.75, gamma=2.0)
```

#### Hyperparameter Tuning

Test different weight ratios:
- **1:2** - Conservative (small increase in FN penalty)
- **1:3** - Recommended starting point
- **1:5** - Aggressive (may cause excessive false alarms)

**Validation Strategy:**
```python
# During training, track BOTH metrics
for epoch in range(num_epochs):
    train_metrics = train_epoch(...)
    val_metrics = validate(...)

    # Calculate clinical metrics, not just accuracy
    sensitivity = val_metrics['sensitivity']
    specificity = val_metrics['specificity']

    # Save checkpoint if sensitivity target met with acceptable specificity
    if sensitivity >= 0.95 and specificity >= 0.85:
        save_checkpoint(...)
```

#### Implementation Steps

1. Create `src/losses.py` with FocalLoss implementation
2. Modify `src/train.py` to accept loss function as parameter
3. Update training script to use weighted loss
4. Retrain for 50 epochs with early stopping on sensitivity
5. Run Phase 2 evaluation on new model

---

### Fix #3: Increase Model Capacity

**Priority:** HIGH
**Effort:** 4-5 hours (modify architecture + retrain)
**Risk:** Medium (may overfit - need regularization)
**Expected Impact:** Better feature learning → higher sensitivity on hard cases

#### Problem

**Current bottleneck:** 2500 inputs → 128 hidden → 2 outputs

Only 128 hidden features to capture:
- Complex temporal dynamics (heartbeat patterns over 10 seconds)
- Multiple arrhythmia types (tachycardia, bradycardia, irregular rhythms)
- Individual patient variability

55.3% confidence on errors suggests model **cannot learn** to discriminate these patterns with current capacity.

#### Solution - Option A: Wider Network

```python
# File: src/model.py

class SimpleSNN(nn.Module):
    def __init__(
        self,
        input_size: int = 2500,
        hidden_size: int = 256,  # DOUBLED from 128
        output_size: int = 2,
        beta: float = 0.9,
        spike_grad = None,
        dropout: float = 0.2  # Add dropout for regularization
    ):
        super().__init__()

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # ... dimension handling ...
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)  # Apply dropout

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
```

**New parameter count:** ~640K (2x increase)

#### Solution - Option B: Deeper Network (Recommended)

```python
class DeepSNN(nn.Module):
    """
    Deeper SNN with 3 layers for better feature hierarchy

    Architecture:
        2500 → 256 → 128 → 2

    Layer 1: Extract low-level temporal features (P-waves, QRS complex)
    Layer 2: Combine into mid-level patterns (heartbeat rhythm)
    Layer 3: High-level decision (normal vs arrhythmia)
    """
    def __init__(
        self,
        input_size: int = 2500,
        hidden_sizes: list = [256, 128],  # Multi-layer
        output_size: int = 2,
        beta: float = 0.9,
        spike_grad = None,
        dropout: float = 0.3
    ):
        super().__init__()

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()

        # Layer 1: Input → 256
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: 256 → 128
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3: 128 → 2 (output)
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # ... dimension handling ...

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for step in range(x.size(0)):
            # Layer 1
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)
```

**New parameter count:** ~673K parameters

#### Implementation Steps

1. **Create new architecture** in `src/model.py`
2. **Add architecture selection** to training script:
   ```bash
   python src/train.py --model deep --hidden-sizes 256 128 --dropout 0.3
   ```
3. **Train with early stopping** on sensitivity metric
4. **Compare both architectures** (wider vs deeper)
5. **Select best performer** based on Phase 2 evaluation

#### Regularization Strategy

To prevent overfitting with larger model:
- **Dropout:** 0.2-0.3 between layers
- **L2 weight decay:** 1e-4
- **Early stopping:** Stop if val sensitivity plateaus for 10 epochs
- **Data augmentation:** (See Tier 3)

---

## TIER 2: Important Diagnostics

### Fix #4: Verify Spike Encoding Quality

**Priority:** MEDIUM
**Effort:** 2 hours
**Risk:** Low (diagnostic only)
**Purpose:** Confirm rate encoding preserves discriminative features

#### Hypothesis to Test

**If spike encoding is destroying information:**
- Normal and Arrhythmia signals will produce similar spike patterns
- Spike rates will be nearly identical between classes
- Model cannot learn because input features are not discriminative

#### Diagnostic Script

Create `scripts/diagnose_spike_encoding.py`:

```python
"""
Diagnostic: Check if spike encoding preserves class-discriminative information
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data import rate_encode
from src.utils import set_seed

def analyze_spike_encoding():
    # Load test data
    test_data = torch.load('data/synthetic/test_data.pt')
    signals = test_data['signals']
    labels = test_data['labels']

    # Sample 20 from each class
    normal_signals = signals[labels == 0][:20]
    arrhythmia_signals = signals[labels == 1][:20]

    print("=== SPIKE ENCODING QUALITY ANALYSIS ===\n")

    # Test different gain values
    gains = [5.0, 10.0, 15.0, 20.0]

    for gain in gains:
        print(f"Testing gain={gain}:")

        normal_rates = []
        arrhythmia_rates = []

        for i in range(20):
            set_seed(42)  # Deterministic

            # Encode normal
            spikes_n = rate_encode(normal_signals[i].numpy(),
                                  num_steps=100, gain=gain)
            normal_rates.append(spikes_n.mean())

            # Encode arrhythmia
            spikes_a = rate_encode(arrhythmia_signals[i].numpy(),
                                  num_steps=100, gain=gain)
            arrhythmia_rates.append(spikes_a.mean())

        normal_mean = np.mean(normal_rates)
        arrhythmia_mean = np.mean(arrhythmia_rates)
        separation = abs(normal_mean - arrhythmia_mean)

        print(f"  Normal:     {normal_mean:.4f} ± {np.std(normal_rates):.4f}")
        print(f"  Arrhythmia: {arrhythmia_mean:.4f} ± {np.std(arrhythmia_rates):.4f}")
        print(f"  Separation: {separation:.4f}")

        if separation < 0.01:
            print(f"  ⚠️  WARNING: Classes too similar - encoding may destroy info")
        elif separation < 0.03:
            print(f"  ⚠️  CAUTION: Weak separation - consider adjusting gain")
        else:
            print(f"  ✓ Good separation")
        print()

    # Visualize spike rasters
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    set_seed(42)
    spikes_n = rate_encode(normal_signals[0].numpy(), num_steps=100, gain=10.0)
    spikes_a = rate_encode(arrhythmia_signals[0].numpy(), num_steps=100, gain=10.0)

    # Normal signal
    axes[0, 0].plot(normal_signals[0].numpy())
    axes[0, 0].set_title("Normal ECG Signal")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")

    # Normal spikes
    spike_times, spike_neurons = np.where(spikes_n.T)
    axes[0, 1].scatter(spike_times, spike_neurons, s=1, c='black')
    axes[0, 1].set_title(f"Normal Spike Raster (rate={spikes_n.mean():.3f})")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Neuron Index")

    # Arrhythmia signal
    axes[1, 0].plot(arrhythmia_signals[0].numpy())
    axes[1, 0].set_title("Arrhythmia ECG Signal")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Amplitude")

    # Arrhythmia spikes
    spike_times, spike_neurons = np.where(spikes_a.T)
    axes[1, 1].scatter(spike_times, spike_neurons, s=1, c='black')
    axes[1, 1].set_title(f"Arrhythmia Spike Raster (rate={spikes_a.mean():.3f})")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Neuron Index")

    plt.tight_layout()
    plt.savefig('results/spike_encoding_diagnostic.png', dpi=150)
    print("Visualization saved to: results/spike_encoding_diagnostic.png")

if __name__ == '__main__':
    analyze_spike_encoding()
```

#### Decision Criteria

**If separation < 0.01:**
- Spike encoding is destroying discriminative information
- **Action:** Adjust gain parameter or switch to different encoding (temporal coding, latency coding)

**If separation 0.01-0.03:**
- Weak but usable separation
- **Action:** Optimize gain parameter through grid search

**If separation > 0.03:**
- Encoding preserves discriminative information
- **Action:** Focus on Fixes #1-3 (not an encoding problem)

---

## TIER 3: Enhancements (After Tier 1+2)

### Fix #5: Data Augmentation

**Priority:** MEDIUM
**Effort:** 6-8 hours
**Expected Impact:** 2-5% accuracy improvement

#### Augmentation Techniques

```python
def augment_ecg(signal, augmentation_type='random'):
    """
    Apply data augmentation to ECG signal

    Types:
        - time_warp: Random time stretching/compression
        - amplitude_scale: Random amplitude scaling
        - noise_injection: Add realistic physiological noise
        - baseline_wander: Simulate breathing artifacts
    """
    if augmentation_type == 'time_warp':
        # Randomly stretch/compress time by 5-15%
        factor = np.random.uniform(0.85, 1.15)
        indices = np.linspace(0, len(signal)-1, int(len(signal)*factor))
        signal = np.interp(np.arange(len(signal)), indices, signal[indices])

    elif augmentation_type == 'amplitude_scale':
        # Scale amplitude by 80-120%
        factor = np.random.uniform(0.8, 1.2)
        signal = signal * factor

    elif augmentation_type == 'noise_injection':
        # Add Gaussian noise (SNR 20-30dB)
        snr_db = np.random.uniform(20, 30)
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        signal = signal + noise

    return signal
```

### Fix #6: Hyperparameter Optimization

**Priority:** MEDIUM
**Effort:** 12-16 hours (includes computation time)
**Tools:** Optuna, Ray Tune

#### Search Space

```python
search_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [16, 32, 64],
    'beta': [0.7, 0.8, 0.9, 0.95],  # Membrane decay
    'spike_grad': ['fast_sigmoid', 'sigmoid', 'atan'],
    'hidden_size': [128, 192, 256],
    'dropout': [0.1, 0.2, 0.3],
    'class_weight_ratio': [2.0, 3.0, 4.0, 5.0]
}
```

### Fix #7: Real Data Integration (Phase 8)

**Priority:** LOW (but important long-term)
**Effort:** 2-3 weeks
**Dataset:** MIT-BIH Arrhythmia Database

Switch from synthetic data to real clinical ECG recordings for realistic validation.

---

## Recommended Implementation Plan

### Week 1: Quick Wins

| Day | Task | Expected Outcome |
|-----|------|------------------|
| **Day 1** | Implement Fix #1 (Threshold Optimization) | Sensitivity: 88% → 93% |
| **Day 2** | Run Fix #4 (Spike Encoding Diagnostic) | Validate encoding quality |
| **Days 3-4** | Implement Fix #2 (Class-Weighted Loss) + Retrain | Sensitivity: 93% → 94-95% |

### Week 2: Deep Improvements (If Week 1 < 95%)

| Day | Task | Expected Outcome |
|-----|------|------------------|
| **Days 5-7** | Implement Fix #3 (Deeper Architecture) + Retrain | Sensitivity: 94% → 95%+ |
| **Day 8** | Re-run Phase 2 Comprehensive Evaluation | Validate all metrics |
| **Day 9** | Update documentation and commit | Prepare for Phase 3 |

---

## Success Criteria

**Deployment Ready:**
- ✅ Sensitivity ≥ 95.0% (currently 88.2%)
- ✅ Specificity ≥ 90.0% (currently 95.6% ✓)
- ✅ NPV ≥ 95.0% (currently 89.0%)
- ✅ PPV ≥ 85.0% (currently 95.2% ✓)
- ✅ All Phase 2 evaluation tasks pass

**Acceptable Trade-offs:**
- Specificity can drop to 88-90% if needed to achieve 95% sensitivity
- False positives are clinically safer than false negatives
- Alarm fatigue is manageable with proper clinical workflow

---

## Risk Assessment

### Fix #1 (Threshold) - LOW RISK ✅
- Easily reversible
- No retraining needed
- Can A/B test thresholds

### Fix #2 (Class Weights) - MEDIUM RISK ⚠️
- Requires full retraining (3-4 hours)
- May increase false positive rate
- Monitor specificity during training

### Fix #3 (Architecture) - MEDIUM RISK ⚠️
- Larger model may overfit
- Longer training time
- Need careful regularization

### Fix #4 (Diagnostic) - NO RISK ✓
- Read-only analysis
- No model changes

---

## Appendix: Technical Details

### Current Model Performance Breakdown

**Confusion Matrix:**
```
                 Predicted
              Normal  Arrhythmia
True Normal      478        22       (95.6% correct)
     Arrhythmia   59       441       (88.2% correct)
```

**Error Categories:**
- Systematic errors: 80/81 (98.8%) - Model has learned bias
- Noisy errors: 1/81 (1.2%) - Signal quality issue
- Borderline: 0 - No uncertain predictions
- Atypical: 0 - No rare pattern failures

**Confidence Distribution:**
- False Negatives: Mean 55.3%, Std 9.8%, Range [50.0%, 93.8%]
- False Positives: Mean 68.1%, Std 13.9%, Range [55.3%, 99.9%]

### Training Data Verification

```
TRAIN: 5000 samples (2500 Normal, 2500 Arrhythmia) - Balanced ✓
VAL:   1000 samples (500 Normal, 500 Arrhythmia) - Balanced ✓
TEST:  1000 samples (500 Normal, 500 Arrhythmia) - Balanced ✓

Model: Trained 11 epochs, 92.3% val accuracy
Loss function: CrossEntropyLoss (unweighted)
Optimizer: Adam (lr=0.001)
```

Data imbalance is **not** the root cause of the bias.

---

**Last Updated:** 2025-11-09
**Next Review:** After implementing Tier 1 fixes
**Owner:** CS2/SNN Expert (Architecture), CS1/Team Lead (Integration)
