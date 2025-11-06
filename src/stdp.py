"""
Spike-Timing-Dependent Plasticity (STDP) Module
===============================================

Owner: CS2 / SNN Expert
Phase: Week 2 (Days 8-14)

This module implements biologically-plausible STDP learning rules for SNNs:
- Classical STDP with exponential kernels
- Homeostatic plasticity for self-stabilization
- Multi-timescale STDP for fast/slow learning

Biological Basis:
    STDP is a fundamental synaptic learning rule observed in biological neurons.
    When a pre-synaptic spike precedes a post-synaptic spike (causality),
    the synapse is strengthened (LTP - Long-Term Potentiation).
    When the order is reversed, the synapse is weakened (LTD - Long-Term Depression).

References:
    - Bi & Poo (1998): "Synaptic modifications in cultured hippocampal neurons"
    - Song et al. (2000): "Competitive Hebbian learning through STDP"
    - Clopath et al. (2010): "Connectivity reflects coding"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Configuration Dataclass
# ============================================

@dataclass
class STDPConfig:
    """
    Configuration for STDP learning parameters

    All parameters are biologically motivated and based on neuroscience literature.
    """
    # Core STDP parameters
    tau_plus: float = 20.0          # LTP time constant (ms) - Bi & Poo (1998)
    tau_minus: float = 20.0         # LTD time constant (ms)
    a_plus: float = 0.01            # LTP learning rate (0.005-0.05 typical)
    a_minus: float = 0.01           # LTD learning rate (often = a_plus)

    # Weight bounds
    w_min: float = 0.0              # Minimum weight (hard bound)
    w_max: float = 1.0              # Maximum weight (hard bound)

    # Practical considerations
    max_lookback: int = 10          # Maximum temporal lookback (timesteps)
    dt: float = 1.0                 # Timestep duration (ms)

    # Homeostatic plasticity (Phase 2)
    use_homeostasis: bool = False
    target_rate: float = 10.0       # Target firing rate (Hz)
    homeostatic_scale: float = 0.001  # Homeostatic adjustment rate

    # Multi-timescale (Phase 3)
    use_multiscale: bool = False
    tau_fast: float = 10.0          # Fast STDP time constant (ms)
    tau_slow: float = 100.0         # Slow STDP time constant (ms)
    alpha_initial: float = 0.8      # Initial fast weight (favor rapid learning)
    alpha_final: float = 0.3        # Final fast weight (favor consolidation)

    # Advanced features
    use_gradient_alignment: bool = False  # Track STDP-backprop alignment
    energy_aware: bool = False      # Modulate STDP by spike energy

    def __post_init__(self):
        """Validate configuration parameters"""
        assert 0 < self.tau_plus <= 1000, "tau_plus must be in (0, 1000] ms"
        assert 0 < self.tau_minus <= 1000, "tau_minus must be in (0, 1000] ms"
        assert 0 <= self.a_plus <= 1.0, "a_plus must be in [0, 1]"
        assert 0 <= self.a_minus <= 1.0, "a_minus must be in [0, 1]"
        assert self.w_min < self.w_max, "w_min must be less than w_max"
        assert self.max_lookback > 0, "max_lookback must be positive"

        if self.use_multiscale:
            assert self.tau_fast < self.tau_slow, "Fast timescale must be shorter than slow"
            assert 0 <= self.alpha_initial <= 1, "alpha_initial must be in [0, 1]"
            assert 0 <= self.alpha_final <= 1, "alpha_final must be in [0, 1]"


# ============================================
# Utility Functions
# ============================================

def calculate_exponential_trace(
    spikes: torch.Tensor,
    tau: float,
    dt: float = 1.0
) -> torch.Tensor:
    """
    Calculate exponential spike trace (filtered spike train)

    This implements the leaky integration of spike events, modeling
    the temporal memory of synaptic efficacy changes.

    Args:
        spikes: Spike train [time_steps, batch, neurons]
        tau: Time constant for exponential decay (ms)
        dt: Timestep duration (ms)

    Returns:
        trace: Exponential trace [time_steps, batch, neurons]

    Mathematical form:
        trace(t+dt) = trace(t) * exp(-dt/tau) + spike(t)
    """
    time_steps, batch, neurons = spikes.shape
    device = spikes.device

    # Exponential decay factor
    decay = torch.exp(torch.tensor(-dt / tau, device=device))

    # Initialize trace
    trace = torch.zeros_like(spikes)
    trace_t = torch.zeros(batch, neurons, device=device)

    # Compute trace recursively
    for t in range(time_steps):
        trace_t = trace_t * decay + spikes[t]
        trace[t] = trace_t

    return trace


def calculate_spike_correlation(
    pre_trace: torch.Tensor,
    post_spikes: torch.Tensor
) -> torch.Tensor:
    """
    Calculate spike correlation for STDP weight updates

    This computes the correlation between pre-synaptic traces and
    post-synaptic spikes, which drives LTP.

    Args:
        pre_trace: Pre-synaptic exponential trace [time_steps, batch, pre_neurons]
        post_spikes: Post-synaptic spikes [time_steps, batch, post_neurons]

    Returns:
        correlation: Correlation matrix [pre_neurons, post_neurons]
    """
    time_steps, batch, _ = pre_trace.shape

    # Sum over time and batch dimensions
    # Outer product: pre_trace^T @ post_spikes
    correlation = torch.einsum('tbi,tbj->ij', pre_trace, post_spikes)

    # Normalize by number of samples
    correlation = correlation / (time_steps * batch)

    return correlation


def clamp_weights(
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
    buffer: float = 0.0
) -> torch.Tensor:
    """
    Clamp weights to valid range with optional buffer

    Args:
        weights: Weight tensor
        w_min: Minimum weight
        w_max: Maximum weight
        buffer: Buffer zone to prevent exact saturation (e.g., 0.05 -> [0.05, 0.95])

    Returns:
        clamped_weights: Weights clamped to [w_min+buffer, w_max-buffer]
    """
    if buffer > 0:
        return torch.clamp(weights, w_min + buffer, w_max - buffer)
    else:
        return torch.clamp(weights, w_min, w_max)


# ============================================
# Classical STDP Layer (Phase 1)
# ============================================

class STDPLayer(nn.Module):
    """
    Classical STDP learning layer with exponential kernels

    This implements the standard STDP rule:
        - LTP (potentiation): Pre-before-post spikes strengthen synapse
        - LTD (depression): Post-before-pre spikes weaken synapse

    Usage:
        stdp_layer = STDPLayer(input_size=2500, output_size=128, config=STDPConfig())
        updated_weights = stdp_layer.apply_stdp(pre_spikes, post_spikes, weights)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[STDPConfig] = None
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.config = config or STDPConfig()

        # Statistics tracking
        self.register_buffer('ltp_events', torch.tensor(0.0))
        self.register_buffer('ltd_events', torch.tensor(0.0))
        self.register_buffer('total_updates', torch.tensor(0.0))
        self.register_buffer('avg_weight_change', torch.tensor(0.0))

        logger.info(f"Initialized STDPLayer: {input_size} → {output_size}")
        logger.info(f"  tau_plus={self.config.tau_plus}ms, tau_minus={self.config.tau_minus}ms")
        logger.info(f"  a_plus={self.config.a_plus}, a_minus={self.config.a_minus}")

    def apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        learning_rate_scale: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply STDP weight update rule

        Args:
            pre_spikes: Pre-synaptic spikes [time_steps, batch, input_size]
            post_spikes: Post-synaptic spikes [time_steps, batch, output_size]
            weights: Current weights [input_size, output_size] or [output_size, input_size]
            learning_rate_scale: Global learning rate multiplier (for annealing)

        Returns:
            updated_weights: New weights after STDP update
            stats: Dictionary with update statistics
        """
        device = weights.device

        # Transpose weights if needed (expect [output, input] for nn.Linear)
        if weights.shape[0] == self.output_size and weights.shape[1] == self.input_size:
            weights = weights  # Correct orientation [output, input]
            transpose_back = False
        elif weights.shape[0] == self.input_size and weights.shape[1] == self.output_size:
            weights = weights.t()  # Transpose to [output, input]
            transpose_back = True
        else:
            raise ValueError(f"Weight shape {weights.shape} incompatible with layer sizes")

        # Calculate exponential traces
        pre_trace_plus = calculate_exponential_trace(
            pre_spikes, self.config.tau_plus, self.config.dt
        )
        post_trace_minus = calculate_exponential_trace(
            post_spikes, self.config.tau_minus, self.config.dt
        )

        # LTP: Post spike causes strengthening based on pre-synaptic trace
        # When post-synaptic neuron fires, look back at recent pre-synaptic activity
        ltp_correlation = calculate_spike_correlation(pre_trace_plus, post_spikes)
        delta_w_ltp = self.config.a_plus * learning_rate_scale * ltp_correlation

        # LTD: Pre spike causes weakening based on post-synaptic trace
        # When pre-synaptic neuron fires, look back at recent post-synaptic activity
        ltd_correlation = calculate_spike_correlation(pre_spikes, post_trace_minus)
        delta_w_ltd = -self.config.a_minus * learning_rate_scale * ltd_correlation

        # Total weight change (transposed to match weight orientation [output, input])
        delta_w = (delta_w_ltp + delta_w_ltd).t()  # Now [output, input]

        # Update weights
        new_weights = weights + delta_w

        # Clamp to valid range
        new_weights = clamp_weights(new_weights, self.config.w_min, self.config.w_max)

        # Update statistics
        with torch.no_grad():
            self.ltp_events += (delta_w_ltp > 0).sum().item()
            self.ltd_events += (delta_w_ltd < 0).sum().item()
            self.total_updates += 1
            self.avg_weight_change = (
                self.avg_weight_change * 0.9 + delta_w.abs().mean() * 0.1
            )

        # Prepare statistics
        stats = {
            'ltp_events': (delta_w_ltp > 0).sum().item(),
            'ltd_events': (delta_w_ltd < 0).sum().item(),
            'ltp_ltd_ratio': (delta_w_ltp > 0).sum().item() / max((delta_w_ltd < 0).sum().item(), 1),
            'avg_delta_w': delta_w.abs().mean().item(),
            'max_delta_w': delta_w.abs().max().item(),
            'weight_mean': new_weights.mean().item(),
            'weight_std': new_weights.std().item(),
            'saturation_ratio': (
                ((new_weights <= self.config.w_min + 0.01).sum() +
                 (new_weights >= self.config.w_max - 0.01).sum()).float() /
                new_weights.numel()
            ).item()
        }

        # Transpose back if needed
        if transpose_back:
            new_weights = new_weights.t()

        return new_weights, stats

    def get_statistics(self) -> Dict[str, float]:
        """Get cumulative STDP statistics"""
        return {
            'total_ltp_events': self.ltp_events.item(),
            'total_ltd_events': self.ltd_events.item(),
            'total_updates': self.total_updates.item(),
            'avg_weight_change': self.avg_weight_change.item()
        }

    def reset_statistics(self):
        """Reset cumulative statistics"""
        self.ltp_events.zero_()
        self.ltd_events.zero_()
        self.total_updates.zero_()
        self.avg_weight_change.zero_()


# ============================================
# Homeostatic STDP (Phase 2)
# ============================================

class HomeostaticSTDP(STDPLayer):
    """
    STDP with homeostatic plasticity for self-stabilization

    Biological Motivation:
        Neurons maintain target firing rates through homeostatic mechanisms.
        This prevents runaway excitation/inhibition and weight saturation.

    Features:
        1. Synaptic Scaling: Multiplicative weight normalization
        2. Target Firing Rate: Adjust weights to maintain desired activity
        3. Heterosynaptic Plasticity: Weak synapses get boosted

    References:
        - Turrigiano & Nelson (2004): "Homeostatic plasticity in the developing nervous system"
        - Zenke et al. (2013): "Synaptic plasticity in neural networks needs homeostasis"
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[STDPConfig] = None
    ):
        super().__init__(input_size, output_size, config)

        # Homeostatic state
        self.register_buffer('neuron_firing_rates', torch.zeros(output_size))
        self.register_buffer('homeostatic_counter', torch.tensor(0))

        logger.info(f"Homeostatic STDP enabled: target_rate={self.config.target_rate} Hz")

    def apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        learning_rate_scale: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply STDP with homeostatic plasticity
        """
        # Standard STDP update
        new_weights, stats = super().apply_stdp(
            pre_spikes, post_spikes, weights, learning_rate_scale
        )

        # Apply homeostatic mechanisms
        if self.config.use_homeostasis:
            new_weights = self._apply_homeostatic_update(
                new_weights, post_spikes
            )

            # Update stats
            stats.update({
                'avg_firing_rate': self.neuron_firing_rates.mean().item(),
                'firing_rate_std': self.neuron_firing_rates.std().item(),
                'rate_deviation': (self.neuron_firing_rates - self.config.target_rate).abs().mean().item()
            })

        return new_weights, stats

    def _apply_homeostatic_update(
        self,
        weights: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply homeostatic plasticity mechanisms

        Args:
            weights: Current weights [output_size, input_size] or [input_size, output_size]
            post_spikes: Post-synaptic spikes [time_steps, batch, output_size]

        Returns:
            weights: Homeostatic-adjusted weights
        """
        # Calculate firing rates (spikes per second)
        time_steps, batch, output_size = post_spikes.shape
        duration_s = time_steps * self.config.dt / 1000.0  # Convert ms to seconds

        current_rates = post_spikes.sum(dim=(0, 1)) / (batch * duration_s)

        # Exponential moving average of firing rates
        alpha = 0.1  # Smoothing factor
        self.neuron_firing_rates = (
            (1 - alpha) * self.neuron_firing_rates + alpha * current_rates
        )

        # Synaptic Scaling: Adjust weights to maintain target firing rate
        # If firing too much → scale down weights
        # If firing too little → scale up weights
        rate_deviation = self.neuron_firing_rates - self.config.target_rate
        scaling_factor = 1.0 - self.config.homeostatic_scale * rate_deviation

        # Apply scaling to weights (handle both orientations)
        if weights.shape[0] == output_size:
            # weights are [output, input]
            weights = weights * scaling_factor.unsqueeze(1)
        else:
            # weights are [input, output]
            weights = weights * scaling_factor.unsqueeze(0)

        # Prevent weak synapses from dying (heterosynaptic plasticity)
        # Add small constant to very weak weights
        min_weight_boost = 0.001
        weights = torch.where(
            weights < 0.05,
            weights + min_weight_boost,
            weights
        )

        # Clamp to valid range
        weights = clamp_weights(weights, self.config.w_min, self.config.w_max)

        self.homeostatic_counter += 1

        return weights


# ============================================
# Multi-Timescale STDP (Phase 3)
# ============================================

class MultiTimescaleSTDP(HomeostaticSTDP):
    """
    STDP with multiple timescales for fast/slow learning

    Biological Motivation:
        Brain learning occurs at multiple timescales:
        - Fast STDP (tau ~10ms): Rapid sensory adaptation
        - Slow STDP (tau ~100ms): Long-term memory consolidation

    This mimics the synaptic tagging and capture hypothesis, where
    synapses have both short-term plasticity (labile) and long-term
    plasticity (consolidated) components.

    References:
        - Frey & Morris (1997): "Synaptic tagging and long-term potentiation"
        - Clopath et al. (2008): "Tag-trigger-consolidation model"
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[STDPConfig] = None
    ):
        super().__init__(input_size, output_size, config)

        # Multi-timescale state
        self.register_buffer('weights_fast', torch.zeros(output_size, input_size))
        self.register_buffer('weights_slow', torch.zeros(output_size, input_size))
        self.register_buffer('alpha', torch.tensor(self.config.alpha_initial))

        logger.info(f"Multi-timescale STDP enabled:")
        logger.info(f"  Fast: tau={self.config.tau_fast}ms, Slow: tau={self.config.tau_slow}ms")
        logger.info(f"  Alpha annealing: {self.config.alpha_initial} → {self.config.alpha_final}")

    def apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        learning_rate_scale: float = 1.0,
        epoch: int = 0,
        max_epochs: int = 70
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply multi-timescale STDP

        Args:
            epoch: Current training epoch (for alpha annealing)
            max_epochs: Total training epochs
        """
        if not self.config.use_multiscale:
            # Fall back to single-timescale
            return super().apply_stdp(pre_spikes, post_spikes, weights, learning_rate_scale)

        device = weights.device

        # Orient weights correctly
        if weights.shape[0] == self.input_size:
            weights = weights.t()  # [output, input]

        # Initialize fast/slow weights if first call
        if self.weights_fast.sum() == 0 and self.weights_slow.sum() == 0:
            self.weights_fast = weights.clone()
            self.weights_slow = weights.clone()

        # Calculate fast STDP update (short tau)
        config_fast = STDPConfig(
            tau_plus=self.config.tau_fast,
            tau_minus=self.config.tau_fast,
            a_plus=self.config.a_plus,
            a_minus=self.config.a_minus
        )

        pre_trace_fast = calculate_exponential_trace(
            pre_spikes, config_fast.tau_plus, config_fast.dt
        )
        post_trace_fast = calculate_exponential_trace(
            post_spikes, config_fast.tau_minus, config_fast.dt
        )

        ltp_fast = calculate_spike_correlation(pre_trace_fast, post_spikes)
        ltd_fast = calculate_spike_correlation(pre_spikes, post_trace_fast)
        delta_w_fast = (config_fast.a_plus * ltp_fast - config_fast.a_minus * ltd_fast).t()

        # Calculate slow STDP update (long tau)
        config_slow = STDPConfig(
            tau_plus=self.config.tau_slow,
            tau_minus=self.config.tau_slow,
            a_plus=self.config.a_plus * 0.5,  # Slower learning for consolidation
            a_minus=self.config.a_minus * 0.5
        )

        pre_trace_slow = calculate_exponential_trace(
            pre_spikes, config_slow.tau_plus, config_slow.dt
        )
        post_trace_slow = calculate_exponential_trace(
            post_spikes, config_slow.tau_minus, config_slow.dt
        )

        ltp_slow = calculate_spike_correlation(pre_trace_slow, post_spikes)
        ltd_slow = calculate_spike_correlation(pre_spikes, post_trace_slow)
        delta_w_slow = (config_slow.a_plus * ltp_slow - config_slow.a_minus * ltd_slow).t()

        # Update fast and slow weight components
        self.weights_fast = self.weights_fast + learning_rate_scale * delta_w_fast
        self.weights_slow = self.weights_slow + learning_rate_scale * delta_w_slow

        # Clamp individual components
        self.weights_fast = clamp_weights(self.weights_fast, self.config.w_min, self.config.w_max)
        self.weights_slow = clamp_weights(self.weights_slow, self.config.w_min, self.config.w_max)

        # Anneal alpha (favor fast early, slow later)
        if max_epochs > 0:
            progress = epoch / max_epochs
            self.alpha = torch.tensor(
                self.config.alpha_initial +
                (self.config.alpha_final - self.config.alpha_initial) * progress
            )

        # Combine fast and slow weights
        new_weights = self.alpha * self.weights_fast + (1 - self.alpha) * self.weights_slow

        # Apply homeostatic mechanisms if enabled
        if self.config.use_homeostasis:
            new_weights = self._apply_homeostatic_update(new_weights, post_spikes)

        # Statistics
        stats = {
            'ltp_fast': (delta_w_fast > 0).sum().item(),
            'ltd_fast': (delta_w_fast < 0).sum().item(),
            'ltp_slow': (delta_w_slow > 0).sum().item(),
            'ltd_slow': (delta_w_slow < 0).sum().item(),
            'alpha': self.alpha.item(),
            'fast_weight_mean': self.weights_fast.mean().item(),
            'slow_weight_mean': self.weights_slow.mean().item(),
            'fast_weight_std': self.weights_fast.std().item(),
            'slow_weight_std': self.weights_slow.std().item(),
            'weight_divergence': (self.weights_fast - self.weights_slow).abs().mean().item()
        }

        return new_weights, stats


# ============================================
# Module Testing
# ============================================

if __name__ == "__main__":
    print("🧪 Testing STDP module...")

    # Test 1: Configuration
    print("\n1. Testing STDPConfig...")
    config = STDPConfig(tau_plus=20.0, a_plus=0.01)
    print(f"   Config created: tau_plus={config.tau_plus}, a_plus={config.a_plus}")

    # Test 2: Exponential trace
    print("\n2. Testing exponential trace...")
    spikes = torch.zeros(10, 2, 5)  # 10 timesteps, 2 batch, 5 neurons
    spikes[2, :, 0] = 1.0  # Neuron 0 spikes at t=2
    spikes[5, :, 2] = 1.0  # Neuron 2 spikes at t=5

    trace = calculate_exponential_trace(spikes, tau=20.0, dt=1.0)
    print(f"   Trace shape: {trace.shape}")
    print(f"   Trace at spike (t=2): {trace[2, 0, 0]:.4f}")
    print(f"   Trace decay (t=3): {trace[3, 0, 0]:.4f}")

    # Test 3: STDP layer
    print("\n3. Testing STDPLayer...")
    stdp = STDPLayer(input_size=5, output_size=3, config=config)

    pre_spikes = torch.rand(10, 2, 5) > 0.8  # Sparse random spikes
    post_spikes = torch.rand(10, 2, 3) > 0.8

    weights = torch.rand(3, 5) * 0.5 + 0.25  # Initialize around 0.5

    new_weights, stats = stdp.apply_stdp(
        pre_spikes.float(), post_spikes.float(), weights
    )

    print(f"   Weight change: {(new_weights - weights).abs().mean():.6f}")
    print(f"   LTP events: {stats['ltp_events']}")
    print(f"   LTD events: {stats['ltd_events']}")
    print(f"   Saturation: {stats['saturation_ratio']:.2%}")

    # Test 4: Homeostatic STDP
    print("\n4. Testing HomeostaticSTDP...")
    config_homeo = STDPConfig(use_homeostasis=True, target_rate=10.0)
    stdp_homeo = HomeostaticSTDP(input_size=5, output_size=3, config=config_homeo)

    new_weights_homeo, stats_homeo = stdp_homeo.apply_stdp(
        pre_spikes.float(), post_spikes.float(), weights
    )

    print(f"   Average firing rate: {stats_homeo['avg_firing_rate']:.2f} Hz")
    print(f"   Rate deviation: {stats_homeo['rate_deviation']:.2f} Hz")

    # Test 5: Multi-timescale STDP
    print("\n5. Testing MultiTimescaleSTDP...")
    config_multi = STDPConfig(use_multiscale=True, tau_fast=10.0, tau_slow=100.0)
    stdp_multi = MultiTimescaleSTDP(input_size=5, output_size=3, config=config_multi)

    new_weights_multi, stats_multi = stdp_multi.apply_stdp(
        pre_spikes.float(), post_spikes.float(), weights, epoch=0, max_epochs=70
    )

    print(f"   Alpha: {stats_multi['alpha']:.3f}")
    print(f"   Fast weight mean: {stats_multi['fast_weight_mean']:.4f}")
    print(f"   Slow weight mean: {stats_multi['slow_weight_mean']:.4f}")
    print(f"   Weight divergence: {stats_multi['weight_divergence']:.6f}")

    print("\n✅ STDP module tests complete!")
    print(f"   Module size: {sum(p.numel() for p in stdp_multi.parameters())} parameters")
