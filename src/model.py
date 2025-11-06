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
from typing import Optional, Tuple, Any, Dict

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
# Hybrid STDP-SNN Architecture (Phase 2)
# ============================================

class HybridSTDP_SNN(SimpleSNN):
    """
    Hybrid SNN with STDP learning for layer 1 and backprop for layer 2

    This implements the biologically-plausible three-phase training strategy:
    - Phase 1 (Epochs 1-20): Pure STDP on layer 1 (unsupervised feature learning)
    - Phase 2 (Epochs 21-50): STDP frozen, backprop on layer 2 (supervised classification)
    - Phase 3 (Epochs 51-70): Fine-tuning with backprop on all layers

    Architecture:
        Input → FC1 + LIF1 (STDP-capable) → FC2 + LIF2 (backprop) → Output

    Key Features:
        - Returns intermediate spike trains for STDP updates
        - Supports homeostatic plasticity (prevents weight saturation)
        - Supports multi-timescale STDP (fast/slow learning)
        - Compatible with existing training infrastructure

    Usage:
        from src.stdp import STDPConfig, MultiTimescaleSTDP

        config = STDPConfig(use_homeostasis=True, use_multiscale=True)
        model = HybridSTDP_SNN(config=config)

        # Phase 1: STDP training
        model.set_stdp_mode(True)
        # ... training loop calls model.apply_stdp_to_layer1()

        # Phase 2: Hybrid training
        model.freeze_layer1()
        model.set_stdp_mode(False)
        # ... backprop training on layer 2

        # Phase 3: Fine-tuning
        model.unfreeze_layer1()
        # ... full backprop training
    """

    def __init__(
        self,
        input_size: int = 2500,
        hidden_size: int = 128,
        output_size: int = 2,
        beta: float = 0.9,
        spike_grad: Optional[Any] = None,
        config: Optional['STDPConfig'] = None
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer size
            output_size: Number of classes
            beta: Membrane potential decay rate
            spike_grad: Surrogate gradient function
            config: STDP configuration (from src.stdp import STDPConfig)
        """
        super().__init__(input_size, hidden_size, output_size, beta, spike_grad)

        # STDP configuration and layer
        if config is not None:
            from src.stdp import MultiTimescaleSTDP
            self.stdp_layer = MultiTimescaleSTDP(
                input_size=input_size,
                output_size=hidden_size,
                config=config
            )
            self.config = config
        else:
            self.stdp_layer = None
            self.config = None

        # Training mode flags
        self.stdp_mode = False  # If True, use STDP for layer 1
        self.layer1_frozen = False  # If True, layer 1 weights frozen

        # Spike history tracking (for STDP)
        self.register_buffer('_input_spikes', None)
        self.register_buffer('_hidden_spikes', None)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional intermediate spike tracking

        Args:
            x: Input tensor [time_steps, batch, features]
            return_intermediate: If True, return input and hidden spikes for STDP

        Returns:
            spikes: Output spikes [time_steps, batch, output_size]
            membrane: Membrane potentials [time_steps, batch, output_size]
            intermediate: Optional tuple (input_spikes, hidden_spikes) for STDP
        """
        # Ensure correct dimension order: [time_steps, batch, features]
        if len(x.shape) == 3 and x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record outputs
        spk1_rec = []  # Hidden layer spikes (for STDP)
        spk2_rec = []  # Output spikes
        mem2_rec = []  # Output membrane potentials

        # Process each time step
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # Stack recordings
        spk2_stacked = torch.stack(spk2_rec)
        mem2_stacked = torch.stack(mem2_rec)

        if return_intermediate or self.stdp_mode:
            spk1_stacked = torch.stack(spk1_rec)
            # Store for STDP update (input is the raw signal, not spikes)
            # For rate encoding, x is already spike-encoded
            self._input_spikes = x.detach()  # [time_steps, batch, input_size]
            self._hidden_spikes = spk1_stacked.detach()  # [time_steps, batch, hidden_size]

            return spk2_stacked, mem2_stacked, (self._input_spikes, self._hidden_spikes)
        else:
            return spk2_stacked, mem2_stacked, None

    def apply_stdp_to_layer1(
        self,
        learning_rate_scale: float = 1.0,
        epoch: int = 0,
        max_epochs: int = 70
    ) -> Optional[Dict[str, float]]:
        """
        Apply STDP weight update to layer 1 (fc1)

        This should be called after forward pass when in STDP mode.
        The forward pass must have been called with return_intermediate=True.

        Args:
            learning_rate_scale: Global learning rate multiplier
            epoch: Current epoch (for multi-timescale alpha annealing)
            max_epochs: Total epochs (for annealing)

        Returns:
            stats: STDP update statistics (LTP/LTD events, weight changes, etc.)
                   Returns None if STDP layer not initialized or no spikes stored
        """
        if self.stdp_layer is None:
            raise RuntimeError("STDP layer not initialized. Pass config to __init__.")

        if self._input_spikes is None or self._hidden_spikes is None:
            raise RuntimeError(
                "No spike history available. Call forward() with return_intermediate=True first."
            )

        # Apply STDP to fc1 weights
        current_weights = self.fc1.weight.data  # [hidden_size, input_size]

        new_weights, stats = self.stdp_layer.apply_stdp(
            pre_spikes=self._input_spikes,
            post_spikes=self._hidden_spikes,
            weights=current_weights,
            learning_rate_scale=learning_rate_scale,
            epoch=epoch,
            max_epochs=max_epochs
        )

        # Update fc1 weights (no_grad since STDP is not part of autograd)
        with torch.no_grad():
            self.fc1.weight.copy_(new_weights)

        # Clear spike history to prevent memory leaks
        self._input_spikes = None
        self._hidden_spikes = None

        return stats

    def set_stdp_mode(self, enabled: bool):
        """
        Enable/disable STDP mode

        When enabled, forward() automatically returns intermediate spikes.
        """
        self.stdp_mode = enabled

    def freeze_layer1(self):
        """
        Freeze layer 1 weights (for Phase 2: backprop only on layer 2)

        This prevents backpropagation from updating STDP-learned features.
        """
        for param in self.fc1.parameters():
            param.requires_grad = False

        # LIF1 has no learnable parameters, so no need to freeze
        self.layer1_frozen = True
        print("✅ Layer 1 (fc1) frozen - STDP weights preserved")

    def unfreeze_layer1(self):
        """
        Unfreeze layer 1 weights (for Phase 3: full fine-tuning)
        """
        for param in self.fc1.parameters():
            param.requires_grad = True

        self.layer1_frozen = False
        print("✅ Layer 1 (fc1) unfrozen - ready for fine-tuning")

    def get_stdp_statistics(self) -> Optional[Dict[str, float]]:
        """
        Get cumulative STDP statistics

        Returns:
            stats: Dictionary with LTP/LTD events, weight changes, etc.
        """
        if self.stdp_layer is not None:
            return self.stdp_layer.get_statistics()
        else:
            return None

    def reset_stdp_statistics(self):
        """Reset STDP cumulative statistics"""
        if self.stdp_layer is not None:
            self.stdp_layer.reset_statistics()


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
