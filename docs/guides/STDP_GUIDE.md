# STDP Implementation Guide

**⚠️ CRITICAL REQUIREMENT**: The problem statement (PS.txt) explicitly requires STDP implementation for biological plausibility.

## What is STDP?

**Spike-Timing-Dependent Plasticity (STDP)** is a biologically-inspired learning rule based on spike timing:
- Weight changes depend on relative timing of pre- and post-synaptic spikes
- **Causality-based**: If pre-spike causes post-spike → strengthen connection (LTP - Long-Term Potentiation)
- **Anti-causality**: If post-spike before pre-spike → weaken connection (LTD - Long-Term Depression)
- Typically unsupervised (no labels needed)

## STDP vs Surrogate Gradient Backpropagation

| Aspect | STDP | Surrogate Gradient Backprop |
|--------|------|----------------------------|
| **Biological Plausibility** | High (brain uses this) | Low (artificial gradient) |
| **Learning Type** | Unsupervised/local | Supervised/global |
| **Task Suitability** | Feature extraction, clustering | Classification, regression |
| **Label Requirement** | No labels needed | Requires labeled data |
| **Convergence Speed** | Slower | Faster |
| **Current Implementation** | TODO | ✅ Used in SimpleSNN |

## When to Use Each

**Use STDP when:**
- Emphasizing biological plausibility for presentation/judges
- Unsupervised feature learning from raw signals
- First layers of deep SNNs (feature extraction)
- Limited labeled data available
- Demonstrating neuromorphic principles

**Use Surrogate Gradient Backprop when:**
- Need high accuracy quickly (hackathon MVP)
- Supervised classification task
- Output/classification layers
- Tight training time constraints

**Best Approach (Hybrid):**
- STDP for early layers (unsupervised feature learning)
- Backprop for final layers (supervised classification)
- Demonstrates biological plausibility + achieves target accuracy

## snnTorch STDP Implementation

### Complete STDP SNN Class

```python
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF

class STDP_SNN(nn.Module):
    """SNN with STDP learning for biological plausibility"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9)

        # STDP parameters
        self.stdp_window = 20.0  # ms, temporal window
        self.a_plus = 0.01       # LTP learning rate
        self.a_minus = 0.01      # LTD learning rate
        self.tau_plus = 20.0     # LTP time constant
        self.tau_minus = 20.0    # LTD time constant

    def stdp_update(self, pre_spikes, post_spikes, weight, dt=1.0):
        """
        Apply STDP weight update rule

        Args:
            pre_spikes: [time_steps, batch, pre_neurons]
            post_spikes: [time_steps, batch, post_neurons]
            weight: [pre_neurons, post_neurons]
            dt: time step in ms

        Returns:
            Updated weight tensor
        """
        time_steps = pre_spikes.size(0)
        batch_size = pre_spikes.size(1)

        # Accumulate weight changes
        delta_w = torch.zeros_like(weight)

        for t in range(time_steps):
            for t_pre in range(max(0, t - int(self.stdp_window)), t + 1):
                # Time difference (post - pre)
                delta_t = (t - t_pre) * dt

                if delta_t >= 0:  # Post after pre → LTP (strengthen)
                    # Exponential STDP curve
                    dw = self.a_plus * torch.exp(-delta_t / self.tau_plus)
                    # Outer product of pre and post spikes
                    delta_w += torch.einsum('bi,bj->ij',
                                           pre_spikes[t_pre],
                                           post_spikes[t]) * dw / batch_size
                else:  # Pre after post → LTD (weaken)
                    dw = -self.a_minus * torch.exp(delta_t / self.tau_minus)
                    delta_w += torch.einsum('bi,bj->ij',
                                           pre_spikes[t_pre],
                                           post_spikes[t]) * dw / batch_size

        # Apply weight update with bounds
        new_weight = weight + delta_w
        return torch.clamp(new_weight, min=0.0, max=1.0)  # Keep weights in [0,1]

    def forward(self, x):
        # Ensure time-first: [time_steps, batch, features]
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_rec = []
        spk2_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk1_rec), torch.stack(spk2_rec)
```

### STDP Training Loop (Unsupervised)

```python
def train_stdp_epoch(model, data_loader, device):
    """Unsupervised STDP training (no labels needed)"""
    model.train()
    total_weight_change = 0.0

    for batch_idx, (data, _) in enumerate(data_loader):  # Ignore labels
        data = data.to(device)

        # Forward pass
        spk1, spk2 = model(data)

        # Apply STDP to first layer
        with torch.no_grad():  # STDP is local, no backprop
            # Get input spikes (data is already spike-encoded)
            input_spikes = data  # [time_steps, batch, input_size]

            # Update first layer weights using STDP
            old_weight = model.fc1.weight.data.clone()
            model.fc1.weight.data = model.stdp_update(
                input_spikes, spk1, model.fc1.weight.data
            )

            # Track weight change magnitude
            weight_change = torch.abs(model.fc1.weight.data - old_weight).sum()
            total_weight_change += weight_change.item()

    avg_weight_change = total_weight_change / len(data_loader)
    return avg_weight_change
```

### Hybrid STDP + Backprop Training (Recommended)

```python
def train_hybrid_epoch(model, data_loader, criterion, optimizer, device):
    """
    Hybrid approach: STDP for layer 1, backprop for layer 2
    Combines biological plausibility with classification accuracy
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        spk1, spk2 = model(data)

        # 1. STDP update for first layer (unsupervised, no backprop)
        with torch.no_grad():
            input_spikes = data
            model.fc1.weight.data = model.stdp_update(
                input_spikes, spk1, model.fc1.weight.data
            )

        # 2. Backprop for second layer (supervised classification)
        optimizer.zero_grad()

        # Use sum of output spikes for classification
        output = spk2.sum(dim=0)  # [batch, output_size]
        loss = criterion(output, targets)

        # Only backprop through layer 2 (freeze layer 1)
        for param in model.fc1.parameters():
            param.requires_grad = False

        loss.backward()
        optimizer.step()

        # Re-enable gradients for layer 1 (in case needed later)
        for param in model.fc1.parameters():
            param.requires_grad = True

        total_loss += loss.item()

    return total_loss / len(data_loader)
```

### STDP Visualization for Demo

```python
def visualize_stdp_learning(model, sample_data, device):
    """
    Generate visualization showing STDP weight evolution
    Great for demo presentation to show biological learning
    """
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        # Capture initial weights
        initial_weights = model.fc1.weight.data.clone()

        # Run forward pass
        spk1, spk2 = model(sample_data.to(device))

        # Apply STDP
        updated_weights = model.stdp_update(
            sample_data, spk1, initial_weights
        )

        # Calculate weight changes
        weight_diff = updated_weights - initial_weights

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Before weights
        im1 = axes[0].imshow(initial_weights.cpu(), cmap='viridis', aspect='auto')
        axes[0].set_title('Weights Before STDP')
        axes[0].set_xlabel('Post-synaptic Neurons')
        axes[0].set_ylabel('Pre-synaptic Neurons')
        plt.colorbar(im1, ax=axes[0])

        # Weight changes (STDP updates)
        im2 = axes[1].imshow(weight_diff.cpu(), cmap='RdBu', aspect='auto',
                            vmin=-0.01, vmax=0.01)
        axes[1].set_title('STDP Weight Changes\n(Red=LTD, Blue=LTP)')
        axes[1].set_xlabel('Post-synaptic Neurons')
        plt.colorbar(im2, ax=axes[1])

        # After weights
        im3 = axes[2].imshow(updated_weights.cpu(), cmap='viridis', aspect='auto')
        axes[2].set_title('Weights After STDP')
        axes[2].set_xlabel('Post-synaptic Neurons')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        return fig
```

## Common STDP Issues

### 1. Weights Saturate to 0 or 1
- **Problem**: All weights go to min/max bounds
- **Solution**: Add weight normalization or homeostatic plasticity

```python
# After STDP update, normalize weights
new_weight = model.stdp_update(pre_spikes, post_spikes, weight)
new_weight = new_weight / new_weight.sum(dim=0, keepdim=True)  # Normalize
```

### 2. Learning Too Slow
- **Problem**: Minimal weight changes over epochs
- **Solution**: Increase learning rates (a_plus, a_minus) or reduce time constants

```python
self.a_plus = 0.05      # Increase from 0.01
self.a_minus = 0.05
self.tau_plus = 10.0    # Decrease from 20.0
```

### 3. No Spike Coincidences
- **Problem**: Pre and post spikes never overlap in time window
- **Solution**: Increase STDP window or adjust spike encoding gain

```python
self.stdp_window = 50.0  # Increase from 20.0
# Or increase spike density in rate_encode()
spike_train = rate_encode(signal, num_steps=100, gain=15.0)  # Higher gain
```

### 4. Memory Issues with Long Sequences
- **Problem**: STDP computation across all timesteps uses too much memory
- **Solution**: Use truncated STDP (only recent spike history)

```python
# Only consider last N timesteps for STDP
max_lookback = 10
for t in range(time_steps):
    for t_pre in range(max(0, t - max_lookback), t + 1):
        # STDP computation
```

## Implementation Roadmap for Hackathon

For **maximum impact** with judges while achieving target accuracy:

### Phase 1 MVP (Days 1-7)
- **Strategy**: Pure surrogate gradient backprop
- **Reason**: Fast convergence, easier debugging, MVP accuracy target (85%)
- **Code**: Current SimpleSNN implementation

### Phase 2 Enhancement (Days 8-14)
- **Strategy**: Add STDP to first layer (hybrid approach)
- **Reason**: Demonstrate biological plausibility while maintaining accuracy
- **Code**: Implement STDP_SNN hybrid model above
- **Deliverable**: STDP weight evolution visualization in demo

### Presentation Strategy
- Show STDP visualization during demo
- Explain biological inspiration (brain-like learning)
- Highlight local learning vs global backprop
- Emphasize energy efficiency of local STDP updates
- Compare STDP feature maps vs random initialization
