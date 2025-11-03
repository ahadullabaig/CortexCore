"""
SNN Model Architectures Module
==============================

Owner: CS2 / SNN Expert

Responsibilities:
- Define SNN architectures
- Implement learning rules (STDP, surrogate gradients)
- Model optimization
- Biological plausibility

Phase: Days 2-30
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from typing import Optional, Tuple, Any

# ============================================
# TODO: Day 2-3 - Basic SNN Model
# ============================================

class SimpleSNN(nn.Module):
    """
    Simple feedforward SNN with LIF neurons

    Architecture:
        Input → FC → LIF → FC → LIF → Output

    TODO:
        - Add more layers
        - Experiment with neuron parameters
        - Implement recurrent connections
    """

    def __init__(
        self,
        input_size: int = 2500,  # 10s * 250Hz
        hidden_size: int = 128,
        output_size: int = 2,
        beta: float = 0.9,
        spike_grad: Optional[Any] = None
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer size
            output_size: Number of classes
            beta: Membrane potential decay rate
            spike_grad: Surrogate gradient function
        """
        super().__init__()

        # Default surrogate gradient
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()

        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor [batch, time_steps, features] or [time_steps, batch, features]

        Returns:
            spikes: Output spikes [time_steps, batch, output_size]
            membrane: Membrane potentials [time_steps, batch, output_size]

        TODO:
            - Add skip connections
            - Implement attention mechanism
        """
        # Ensure correct dimension order: [time_steps, batch, features]
        if len(x.shape) == 3 and x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record outputs
        spk2_rec = []
        mem2_rec = []

        # Process each time step
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


# ============================================
# TODO: Day 5-7 - Enhanced SNN
# ============================================

class DeepSNN(nn.Module):
    """
    Deeper SNN with multiple hidden layers

    TODO:
        - Implement this during Week 1
        - Add batch normalization
        - Add dropout for regularization
    """

    def __init__(self, input_size: int = 2500, num_classes: int = 2):
        super().__init__()
        # TODO: Implement architecture
        raise NotImplementedError("Week 1 task")


# ============================================
# TODO: Day 8-14 - Hybrid SNN-ANN
# ============================================

class HybridSNN_ANN(nn.Module):
    """
    Hybrid architecture combining SNN and ANN

    Architecture:
        Input → CNN (ANN) → Feature extraction → SNN → Classification

    TODO:
        - Implement CNN feature extractor
        - Add SNN classifier head
        - Design fusion mechanism
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # TODO: Implement during Week 2
        raise NotImplementedError("Week 2 task")


# ============================================
# TODO: Day 15+ - Advanced Architectures
# ============================================

class RecurrentSNN(nn.Module):
    """Recurrent SNN with temporal dynamics"""

    def __init__(self):
        super().__init__()
        # TODO: Week 3 task
        raise NotImplementedError("Week 3 task")


class AttentionSNN(nn.Module):
    """SNN with attention mechanisms"""

    def __init__(self):
        super().__init__()
        # TODO: Week 3 task
        raise NotImplementedError("Week 3 task")


# ============================================
# Learning Rules
# ============================================

class STDP:
    """
    Spike-Timing-Dependent Plasticity learning rule

    TODO: Day 8-10
        - Implement classical STDP
        - Add reward modulation
        - Test on simple tasks
    """

    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """TODO: Implement STDP weight update"""
        raise NotImplementedError("Week 2 task")


# ============================================
# Model Utilities
# ============================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_sparsity(spikes: torch.Tensor) -> float:
    """Calculate spike sparsity (percentage of inactive neurons)"""
    return (1 - spikes.sum() / spikes.numel()).item()


def measure_energy_efficiency(spikes: torch.Tensor, ann_activations: Optional[torch.Tensor] = None) -> dict:
    """
    Estimate energy efficiency compared to ANN

    Args:
        spikes: SNN spikes
        ann_activations: ANN activations for comparison

    Returns:
        Energy metrics

    TODO:
        - Implement realistic energy model
        - Add neuromorphic hardware estimates
    """
    n_spikes = spikes.sum().item()
    n_neurons = spikes.numel()

    metrics = {
        'total_spikes': n_spikes,
        'total_neurons': n_neurons,
        'sparsity': (1 - n_spikes / n_neurons) * 100,
        'spike_rate': n_spikes / n_neurons
    }

    if ann_activations is not None:
        ann_ops = ann_activations.numel()
        metrics['ops_reduction'] = (1 - n_spikes / ann_ops) * 100

    return metrics


# ============================================
# Example Usage & Testing
# ============================================

if __name__ == "__main__":
    print("🧪 Testing model module...")

    # Create model
    print("\n1. Creating SimpleSNN...")
    model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
    print(f"   Parameters: {count_parameters(model):,}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    time_steps = 100
    input_features = 2500

    x = torch.randn(time_steps, batch_size, input_features)
    spikes, membrane = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output spikes shape: {spikes.shape}")
    print(f"   Membrane shape: {membrane.shape}")

    # Calculate metrics
    print("\n3. Calculating metrics...")
    sparsity = calculate_sparsity(spikes)
    print(f"   Sparsity: {sparsity*100:.2f}%")

    energy_metrics = measure_energy_efficiency(spikes)
    print(f"   Total spikes: {energy_metrics['total_spikes']:.0f}")
    print(f"   Spike rate: {energy_metrics['spike_rate']:.4f}")

    # Test with different inputs
    print("\n4. Testing gradient flow...")
    x.requires_grad = True
    spikes, _ = model(x)
    loss = spikes.sum()
    loss.backward()
    print(f"   Gradient exists: {x.grad is not None}")
    print(f"   Gradient shape: {x.grad.shape if x.grad is not None else None}")

    print("\n✅ Model module working!")
