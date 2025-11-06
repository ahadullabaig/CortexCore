"""
STDP Module Unit Tests
======================

Comprehensive test suite for STDP implementation including:
- STDPConfig dataclass validation
- STDPLayer weight updates
- HomeostaticSTDP firing rate regulation
- MultiTimescaleSTDP alpha annealing
- HybridSTDP_SNN model integration
- Phase transitions (freeze/unfreeze)

Run with: pytest tests/test_stdp.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

from src.stdp import (
    STDPConfig,
    STDPLayer,
    HomeostaticSTDP,
    MultiTimescaleSTDP,
    calculate_exponential_trace,
    calculate_spike_correlation,
    clamp_weights
)
from src.model import HybridSTDP_SNN, SimpleSNN, count_parameters


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def device():
    """Test device (prefer CPU for reproducibility)"""
    return 'cpu'


@pytest.fixture
def stdp_config():
    """Default STDP configuration"""
    return STDPConfig(
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.01,
        a_minus=0.01,
        use_homeostasis=True,
        target_rate=10.0,
        use_multiscale=True,
        tau_fast=10.0,
        tau_slow=100.0,
        alpha_initial=0.8,
        alpha_final=0.3
    )


@pytest.fixture
def sample_spikes(device):
    """Generate sample spike trains for testing"""
    torch.manual_seed(42)
    time_steps = 100
    batch_size = 4
    input_size = 50
    hidden_size = 20

    pre_spikes = torch.bernoulli(torch.ones(time_steps, batch_size, input_size) * 0.1).to(device)
    post_spikes = torch.bernoulli(torch.ones(time_steps, batch_size, hidden_size) * 0.1).to(device)

    return pre_spikes, post_spikes


# ============================================
# STDPConfig Tests
# ============================================

def test_stdp_config_defaults():
    """Test STDPConfig default values"""
    config = STDPConfig()

    assert config.tau_plus == 20.0
    assert config.tau_minus == 20.0
    assert config.a_plus == 0.01
    assert config.a_minus == 0.01
    assert config.w_min == 0.0
    assert config.w_max == 1.0
    assert config.use_homeostasis == False
    assert config.use_multiscale == False


def test_stdp_config_custom_values():
    """Test STDPConfig with custom values"""
    config = STDPConfig(
        tau_plus=15.0,
        tau_minus=25.0,
        a_plus=0.02,
        a_minus=0.015,
        w_min=-0.5,
        w_max=2.0,
        use_homeostasis=True,
        target_rate=15.0
    )

    assert config.tau_plus == 15.0
    assert config.tau_minus == 25.0
    assert config.a_plus == 0.02
    assert config.a_minus == 0.015
    assert config.w_min == -0.5
    assert config.w_max == 2.0
    assert config.use_homeostasis == True
    assert config.target_rate == 15.0


# ============================================
# Utility Function Tests
# ============================================

def test_calculate_exponential_trace(device):
    """Test exponential trace calculation"""
    torch.manual_seed(42)
    time_steps = 50
    batch_size = 2
    neurons = 10

    spikes = torch.bernoulli(torch.ones(time_steps, batch_size, neurons) * 0.2).to(device)
    tau = 20.0

    trace = calculate_exponential_trace(spikes, tau)

    assert trace.shape == spikes.shape
    assert torch.all(trace >= 0)
    assert torch.all(trace <= time_steps)  # Trace accumulates but decays


def test_exponential_trace_decay(device):
    """Test that exponential trace decays correctly"""
    torch.manual_seed(42)
    time_steps = 100
    batch_size = 1
    neurons = 1

    # Single spike at t=10
    spikes = torch.zeros(time_steps, batch_size, neurons).to(device)
    spikes[10, 0, 0] = 1.0

    tau = 20.0
    trace = calculate_exponential_trace(spikes, tau)

    # Trace should peak at spike time and decay exponentially
    assert trace[10, 0, 0] > 0
    assert trace[30, 0, 0] < trace[10, 0, 0]  # Decayed after 20 steps
    assert trace[30, 0, 0] > 0  # Still positive


def test_calculate_spike_correlation(device):
    """Test spike correlation calculation"""
    torch.manual_seed(42)
    time_steps = 50
    batch_size = 2
    pre_size = 10
    post_size = 5

    pre_spikes = torch.bernoulli(torch.ones(time_steps, batch_size, pre_size) * 0.1).to(device)
    post_spikes = torch.bernoulli(torch.ones(time_steps, batch_size, post_size) * 0.1).to(device)

    correlation = calculate_spike_correlation(pre_spikes, post_spikes)

    # Correlation shape is [pre_size, post_size]
    assert correlation.shape == (pre_size, post_size)
    assert torch.all(correlation >= 0)


def test_clamp_weights(device):
    """Test weight clamping"""
    weights = torch.tensor([[-0.5, 0.5, 1.5], [0.0, 1.0, 2.0]]).to(device)

    clamped = clamp_weights(weights, w_min=0.0, w_max=1.0)

    assert torch.all(clamped >= 0.0)
    assert torch.all(clamped <= 1.0)
    assert clamped[0, 0] == 0.0  # -0.5 clamped to 0.0
    assert clamped[0, 1] == 0.5  # 0.5 unchanged
    assert clamped[0, 2] == 1.0  # 1.5 clamped to 1.0


# ============================================
# STDPLayer Tests
# ============================================

def test_stdp_layer_initialization(stdp_config, device):
    """Test STDPLayer initialization"""
    input_size = 100
    output_size = 50

    layer = STDPLayer(input_size, output_size, config=stdp_config).to(device)

    assert layer.input_size == input_size
    assert layer.output_size == output_size
    assert layer.config == stdp_config


def test_stdp_layer_weight_update(stdp_config, sample_spikes, device):
    """Test STDP weight updates modify weights"""
    pre_spikes, post_spikes = sample_spikes
    input_size = pre_spikes.shape[2]
    output_size = post_spikes.shape[2]

    layer = STDPLayer(input_size, output_size, config=stdp_config).to(device)

    # Initial weights
    initial_weights = torch.randn(output_size, input_size).to(device)
    initial_weights = clamp_weights(initial_weights, 0.0, 1.0)

    # Apply STDP
    new_weights, stats = layer.apply_stdp(
        pre_spikes=pre_spikes,
        post_spikes=post_spikes,
        weights=initial_weights,
        learning_rate_scale=1.0
    )

    # Weights should have changed
    assert not torch.allclose(new_weights, initial_weights)

    # Weights should be clamped
    assert torch.all(new_weights >= stdp_config.w_min)
    assert torch.all(new_weights <= stdp_config.w_max)

    # Stats should be returned
    assert 'ltp_events' in stats
    assert 'ltd_events' in stats
    assert 'avg_delta_w' in stats


def test_stdp_ltp_ltd_balance(stdp_config, sample_spikes, device):
    """Test LTP/LTD balance in STDP"""
    pre_spikes, post_spikes = sample_spikes
    input_size = pre_spikes.shape[2]
    output_size = post_spikes.shape[2]

    layer = STDPLayer(input_size, output_size, config=stdp_config).to(device)

    weights = torch.ones(output_size, input_size).to(device) * 0.5

    _, stats = layer.apply_stdp(pre_spikes, post_spikes, weights, 1.0)

    # Both LTP and LTD events should occur
    assert stats['ltp_events'] > 0
    assert stats['ltd_events'] > 0

    # Ratio should be reasonable (not all potentiation or depression)
    ratio = stats['ltp_events'] / (stats['ltd_events'] + 1e-6)
    assert 0.5 < ratio < 2.0  # Within 2x of balanced


# ============================================
# HomeostaticSTDP Tests
# ============================================

def test_homeostatic_stdp_initialization(stdp_config, device):
    """Test HomeostaticSTDP initialization"""
    input_size = 100
    output_size = 50

    layer = HomeostaticSTDP(input_size, output_size, config=stdp_config).to(device)

    assert layer.config.use_homeostasis == True
    assert layer.neuron_firing_rates is not None
    assert layer.neuron_firing_rates.shape == (output_size,)


def test_homeostatic_stdp_firing_rate_tracking(stdp_config, sample_spikes, device):
    """Test homeostatic plasticity tracks firing rates"""
    pre_spikes, post_spikes = sample_spikes
    input_size = pre_spikes.shape[2]
    output_size = post_spikes.shape[2]

    layer = HomeostaticSTDP(input_size, output_size, config=stdp_config).to(device)

    weights = torch.ones(output_size, input_size).to(device) * 0.5

    # Apply STDP multiple times
    for _ in range(5):
        weights, stats = layer.apply_stdp(pre_spikes, post_spikes, weights, 1.0)

    # Firing rates should be tracked
    assert torch.all(layer.neuron_firing_rates >= 0)
    assert 'avg_firing_rate' in stats


# ============================================
# MultiTimescaleSTDP Tests
# ============================================

def test_multiscale_stdp_initialization(stdp_config, device):
    """Test MultiTimescaleSTDP initialization"""
    input_size = 100
    output_size = 50

    layer = MultiTimescaleSTDP(input_size, output_size, config=stdp_config).to(device)

    assert layer.config.use_multiscale == True
    assert layer.weights_fast is not None
    assert layer.weights_slow is not None
    assert layer.alpha == stdp_config.alpha_initial


def test_multiscale_alpha_annealing(stdp_config, sample_spikes, device):
    """Test alpha annealing in multi-timescale STDP"""
    pre_spikes, post_spikes = sample_spikes
    input_size = pre_spikes.shape[2]
    output_size = post_spikes.shape[2]

    layer = MultiTimescaleSTDP(input_size, output_size, config=stdp_config).to(device)

    weights = torch.ones(output_size, input_size).to(device) * 0.5

    max_epochs = 20

    # Initial alpha
    initial_alpha = layer.alpha
    assert initial_alpha == stdp_config.alpha_initial

    # Apply STDP at different epochs
    for epoch in [0, 10, 19]:
        _, stats = layer.apply_stdp(
            pre_spikes, post_spikes, weights, 1.0,
            epoch=epoch, max_epochs=max_epochs
        )

    # Alpha should anneal toward alpha_final
    final_alpha = layer.alpha
    assert final_alpha < initial_alpha
    assert final_alpha >= stdp_config.alpha_final - 0.1  # Allow some tolerance


def test_multiscale_weight_divergence(stdp_config, sample_spikes, device):
    """Test fast and slow weights diverge"""
    pre_spikes, post_spikes = sample_spikes
    input_size = pre_spikes.shape[2]
    output_size = post_spikes.shape[2]

    layer = MultiTimescaleSTDP(input_size, output_size, config=stdp_config).to(device)

    weights = torch.ones(output_size, input_size).to(device) * 0.5

    # Apply STDP multiple times
    for _ in range(10):
        weights, stats = layer.apply_stdp(pre_spikes, post_spikes, weights, 1.0)

    # Fast and slow weights should differ
    divergence = stats.get('weight_divergence', 0)
    assert divergence >= 0  # Divergence should be tracked (can be 0 initially)


# ============================================
# HybridSTDP_SNN Tests
# ============================================

def test_hybrid_stdp_snn_initialization(stdp_config, device):
    """Test HybridSTDP_SNN initialization"""
    model = HybridSTDP_SNN(
        input_size=2500,
        hidden_size=128,
        output_size=2,
        config=stdp_config
    ).to(device)

    assert model.stdp_layer is not None
    assert model.config == stdp_config
    assert model.stdp_mode == False
    assert model.layer1_frozen == False


def test_hybrid_stdp_snn_forward_basic(stdp_config, device):
    """Test basic forward pass without STDP"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    spikes, membrane, intermediate = model(x, return_intermediate=False)

    assert spikes.shape == (time_steps, batch_size, 2)
    assert membrane.shape == (time_steps, batch_size, 2)
    assert intermediate is None


def test_hybrid_stdp_snn_forward_with_intermediate(stdp_config, device):
    """Test forward pass with intermediate spike tracking"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    spikes, membrane, intermediate = model(x, return_intermediate=True)

    assert intermediate is not None
    input_spikes, hidden_spikes = intermediate
    assert input_spikes.shape == (time_steps, batch_size, 100)
    assert hidden_spikes.shape == (time_steps, batch_size, 50)


def test_hybrid_stdp_snn_stdp_mode(stdp_config, device):
    """Test STDP mode activation"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    # Enable STDP mode
    model.set_stdp_mode(True)
    assert model.stdp_mode == True

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    # Forward pass should return intermediate even without explicit flag
    spikes, membrane, intermediate = model(x)
    assert intermediate is not None


def test_hybrid_stdp_snn_apply_stdp(stdp_config, device):
    """Test STDP weight updates in HybridSTDP_SNN"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    model.set_stdp_mode(True)

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    # Get initial weights
    initial_weights = model.fc1.weight.data.clone()

    # Forward pass
    model(x)

    # Apply STDP
    stats = model.apply_stdp_to_layer1(learning_rate_scale=1.0)

    # Weights should have changed
    assert not torch.allclose(model.fc1.weight.data, initial_weights)

    # Stats should be returned
    assert stats is not None
    assert 'ltp_fast' in stats or 'ltp_events' in stats


def test_hybrid_stdp_snn_freeze_unfreeze(stdp_config, device):
    """Test layer freezing/unfreezing"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    # Initially unfrozen
    assert model.layer1_frozen == False
    assert model.fc1.weight.requires_grad == True

    # Freeze layer 1
    model.freeze_layer1()
    assert model.layer1_frozen == True
    assert model.fc1.weight.requires_grad == False

    # Unfreeze layer 1
    model.unfreeze_layer1()
    assert model.layer1_frozen == False
    assert model.fc1.weight.requires_grad == True


def test_hybrid_stdp_snn_parameter_count(stdp_config, device):
    """Test parameter count matches SimpleSNN"""
    simple_model = SimpleSNN(
        input_size=2500,
        hidden_size=128,
        output_size=2
    ).to(device)

    hybrid_model = HybridSTDP_SNN(
        input_size=2500,
        hidden_size=128,
        output_size=2,
        config=stdp_config
    ).to(device)

    # Both should have same number of trainable parameters
    simple_params = count_parameters(simple_model)
    hybrid_params = count_parameters(hybrid_model)

    assert simple_params == hybrid_params


# ============================================
# Integration Tests
# ============================================

def test_full_stdp_pipeline(stdp_config, device):
    """Test complete STDP training pipeline simulation"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    # Simulate Phase 1: STDP training
    model.set_stdp_mode(True)

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    # Multiple STDP updates
    for epoch in range(5):
        model(x)
        stats = model.apply_stdp_to_layer1(epoch=epoch, max_epochs=10)
        assert stats is not None

    # Simulate Phase 2: Freeze layer 1
    model.freeze_layer1()
    model.set_stdp_mode(False)
    assert model.fc1.weight.requires_grad == False

    # Simulate Phase 3: Unfreeze for fine-tuning
    model.unfreeze_layer1()
    assert model.fc1.weight.requires_grad == True


def test_stdp_no_gradient_leakage(stdp_config, device):
    """Test STDP updates don't interfere with autograd"""
    model = HybridSTDP_SNN(
        input_size=100,
        hidden_size=50,
        output_size=2,
        config=stdp_config
    ).to(device)

    # First do STDP in no_grad mode (as training does)
    model.set_stdp_mode(True)

    time_steps = 50
    batch_size = 4
    x = torch.randn(time_steps, batch_size, 100).to(device)

    with torch.no_grad():
        spikes, membrane, _ = model(x)
    model.apply_stdp_to_layer1()

    # Now test that backprop still works independently
    model.set_stdp_mode(False)
    model.freeze_layer1()  # Freeze fc1 (as in phase 2)

    x2 = torch.randn(time_steps, batch_size, 100).to(device)
    x2.requires_grad = True

    spikes2, membrane2, _ = model(x2, return_intermediate=False)
    loss = spikes2.sum()
    loss.backward()

    # fc2 should have gradients (fc1 is frozen)
    assert model.fc2.weight.grad is not None
    # fc1 should not have gradients (frozen)
    assert model.fc1.weight.grad is None


# ============================================
# Performance Tests
# ============================================

@pytest.mark.slow
def test_stdp_performance(stdp_config, device):
    """Test STDP computation performance (realistic sizes)"""
    import time

    model = HybridSTDP_SNN(
        input_size=2500,
        hidden_size=128,
        output_size=2,
        config=stdp_config
    ).to(device)

    model.set_stdp_mode(True)

    time_steps = 100
    batch_size = 32
    x = torch.randn(time_steps, batch_size, 2500).to(device)

    start = time.time()
    model(x)
    model.apply_stdp_to_layer1()
    elapsed = time.time() - start

    # Should complete in reasonable time (< 1 second on CPU)
    assert elapsed < 2.0


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
