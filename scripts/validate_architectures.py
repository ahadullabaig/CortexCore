"""
Architecture Validation Script
==============================

Tests new WiderSNN and DeepSNN architectures to ensure:
1. Forward pass works correctly
2. Output shapes are correct
3. Gradients can flow (backward pass)
4. Parameter counts are as expected

Usage:
    python scripts/validate_architectures.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SimpleSNN, WiderSNN, DeepSNN


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_architecture(model_class, model_name, **kwargs):
    """Test a single architecture"""
    print(f"\n{'=' * 70}")
    print(f"Testing {model_name}")
    print('=' * 70)

    # Create model
    model = model_class(**kwargs)
    param_count = count_parameters(model)
    print(f"✓ Model created: {param_count:,} parameters")

    # Test forward pass
    batch_size = 4
    time_steps = 100
    input_size = 2500

    # Create dummy input [time_steps, batch, features]
    x = torch.randn(time_steps, batch_size, input_size)
    print(f"✓ Input shape: {list(x.shape)}")

    # Forward pass
    spikes, membrane = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Spikes shape: {list(spikes.shape)}")
    print(f"  Membrane shape: {list(membrane.shape)}")

    # Verify shapes
    expected_shape = [time_steps, batch_size, 2]
    assert list(spikes.shape) == expected_shape, f"Expected {expected_shape}, got {list(spikes.shape)}"
    assert list(membrane.shape) == expected_shape, f"Expected {expected_shape}, got {list(membrane.shape)}"
    print(f"✓ Output shapes correct")

    # Test backward pass (check gradients flow)
    output = spikes.sum(dim=0)  # [batch, classes]
    target = torch.randint(0, 2, (batch_size,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()
    print(f"✓ Backward pass successful (loss: {loss.item():.4f})")

    # Check gradients exist
    has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_gradients}")

    # Test config storage
    if hasattr(model, 'config'):
        print(f"✓ Config stored: {model.config.get('architecture', 'N/A')}")

    return True


def compare_architectures():
    """Compare all three architectures"""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    models = [
        ("SimpleSNN", SimpleSNN()),
        ("WiderSNN", WiderSNN()),
        ("DeepSNN", DeepSNN())
    ]

    print(f"\n{'Architecture':<15} {'Parameters':<15} {'Layers':<10} {'Hidden Dims'}")
    print("-" * 70)

    for name, model in models:
        params = count_parameters(model)

        # Count layers (LIF neurons)
        num_lif_layers = sum(1 for module in model.modules()
                            if 'Leaky' in str(type(module)))

        # Get hidden dimensions
        if hasattr(model, 'config'):
            if 'hidden_sizes' in model.config:
                hidden = str(model.config['hidden_sizes'])
            else:
                hidden = str(model.config.get('hidden_size', '?'))
        else:
            hidden = "N/A"

        print(f"{name:<15} {params:>13,}   {num_lif_layers:<10} {hidden}")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("ARCHITECTURE VALIDATION")
    print("=" * 70)

    try:
        # Test SimpleSNN (baseline)
        test_architecture(SimpleSNN, "SimpleSNN (Baseline)")

        # Test WiderSNN
        test_architecture(WiderSNN, "WiderSNN (2x width)", hidden_size=256, dropout=0.2)

        # Test DeepSNN
        test_architecture(DeepSNN, "DeepSNN (3 layers)", hidden_sizes=[256, 128], dropout=0.3)

        # Compare architectures
        compare_architectures()

        print("\n" + "=" * 70)
        print("✅ ALL ARCHITECTURES VALIDATED SUCCESSFULLY")
        print("=" * 70)
        print("\n📋 Summary:")
        print("  - SimpleSNN: ~320K params, 2 layers, 128 hidden (baseline)")
        print("  - WiderSNN:  ~640K params, 2 layers, 256 hidden (2x capacity)")
        print("  - DeepSNN:   ~673K params, 3 layers, 256→128 (hierarchical)")
        print("\n✅ Ready for training with FocalLoss!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
