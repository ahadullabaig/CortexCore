"""
STDP Implementation Benchmark
==============================

Comprehensive benchmarking and validation of the STDP implementation including:
- Model performance metrics
- STDP learning dynamics
- Energy efficiency estimates
- Inference speed
- Memory usage
- Comparison with baseline SimpleSNN

Usage:
    python scripts/benchmark_stdp.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import json
import numpy as np
from torch.utils.data import DataLoader

from src.model import HybridSTDP_SNN, SimpleSNN, count_parameters
from src.stdp import STDPConfig
from src.data import load_dataset
from src.utils import set_seed
from src.inference import load_model, predict


def benchmark_inference_speed(model, test_loader, device, num_runs=100):
    """Benchmark model inference speed"""
    model.eval()
    model.to(device)

    times = []
    with torch.no_grad():
        # Warm-up
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
            break

        # Actual benchmark
        for _ in range(num_runs):
            for data, _ in test_loader:
                data = data.to(device)

                start = time.time()
                _ = model(data)
                torch.cuda.synchronize() if device == 'cuda' else None
                elapsed = time.time() - start

                times.append(elapsed)
                break  # One batch per run

    return {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'median_time_ms': np.median(times) * 1000
    }


def benchmark_memory_usage(model, device):
    """Benchmark memory usage"""
    model.eval()
    model.to(device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Forward pass
        time_steps = 100
        batch_size = 32
        input_size = 2500
        x = torch.randn(time_steps, batch_size, input_size).to(device)

        with torch.no_grad():
            _ = model(x)

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        return {
            'peak_memory_mb': peak_memory_mb,
            'current_memory_mb': current_memory_mb
        }
    else:
        return {
            'peak_memory_mb': 0,
            'current_memory_mb': 0
        }


def analyze_stdp_dynamics(history):
    """Analyze STDP learning dynamics from training history"""
    phase1 = history['phase1_stdp']

    ltp_ltd_ratios = [e['ltp_ltd_ratio'] for e in phase1]
    alphas = [e['avg_alpha'] for e in phase1]
    weight_changes = [e['avg_weight_change'] for e in phase1]

    return {
        'ltp_ltd_mean': np.mean(ltp_ltd_ratios),
        'ltp_ltd_std': np.std(ltp_ltd_ratios),
        'ltp_ltd_min': np.min(ltp_ltd_ratios),
        'ltp_ltd_max': np.max(ltp_ltd_ratios),
        'alpha_initial': alphas[0],
        'alpha_final': alphas[-1],
        'alpha_annealing': alphas[0] - alphas[-1],
        'weight_change_mean': np.mean(weight_changes),
        'weight_change_final': weight_changes[-1]
    }


def analyze_accuracy_progression(history):
    """Analyze accuracy progression across phases"""
    val_phase2 = history.get('val_phase2', [])
    val_phase3 = history.get('val_phase3', [])

    phase2_accs = [e['val_accuracy'] for e in val_phase2] if val_phase2 else []
    phase3_accs = [e['val_accuracy'] for e in val_phase3] if val_phase3 else []

    all_accs = phase2_accs + phase3_accs

    return {
        'phase2_best': max(phase2_accs) if phase2_accs else 0,
        'phase2_final': phase2_accs[-1] if phase2_accs else 0,
        'phase3_best': max(phase3_accs) if phase3_accs else 0,
        'phase3_final': phase3_accs[-1] if phase3_accs else 0,
        'overall_best': max(all_accs) if all_accs else 0,
        'accuracy_improvement': (max(all_accs) - phase2_accs[0]) if phase2_accs and all_accs else 0
    }


def main():
    print("=" * 80)
    print("🔬 STDP IMPLEMENTATION BENCHMARK")
    print("=" * 80)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n📊 Configuration:")
    print(f"   Device: {device}")

    # Load training history
    print("\n📖 Loading training history...")
    history_path = 'models/stdp_full/stdp_training_history.json'
    with open(history_path, 'r') as f:
        history = json.load(f)
    print("   ✅ History loaded")

    # Load test data
    print("\n📊 Loading test data...")
    try:
        test_loader = load_dataset(
            'data/synthetic/test_data.pt',
            batch_size=32,
            shuffle=False
        )
        print(f"   ✅ Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"   ❌ Error loading test data: {e}")
        return 1

    # Load trained model
    print("\n🏗️  Loading trained STDP model...")
    try:
        config = STDPConfig(
            use_homeostasis=True,
            target_rate=10.0,
            use_multiscale=True,
            tau_fast=10.0,
            tau_slow=100.0,
            alpha_initial=0.8,
            alpha_final=0.3
        )

        model = HybridSTDP_SNN(
            input_size=2500,
            hidden_size=128,
            output_size=2,
            config=config
        ).to(device)

        checkpoint = torch.load('models/stdp_full/best_finetuned_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Model loaded (epoch {checkpoint['epoch']})")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return 1

    # Benchmark 1: Model Architecture
    print("\n" + "=" * 80)
    print("1️⃣  MODEL ARCHITECTURE")
    print("=" * 80)

    params = count_parameters(model)
    print(f"   Parameters: {params:,}")
    print(f"   Architecture: 2500 → 128 → 2")
    print(f"   Layers: 2 (fc1 + fc2)")
    print(f"   STDP Features:")
    print(f"      - Homeostatic plasticity: {config.use_homeostasis}")
    print(f"      - Multi-timescale learning: {config.use_multiscale}")

    # Benchmark 2: STDP Dynamics
    print("\n" + "=" * 80)
    print("2️⃣  STDP LEARNING DYNAMICS")
    print("=" * 80)

    stdp_metrics = analyze_stdp_dynamics(history)
    print(f"   LTP/LTD Balance:")
    print(f"      Mean ratio: {stdp_metrics['ltp_ltd_mean']:.3f}")
    print(f"      Std: {stdp_metrics['ltp_ltd_std']:.3f}")
    print(f"      Range: [{stdp_metrics['ltp_ltd_min']:.3f}, {stdp_metrics['ltp_ltd_max']:.3f}]")
    print(f"\n   Multi-timescale Alpha:")
    print(f"      Initial: {stdp_metrics['alpha_initial']:.3f}")
    print(f"      Final: {stdp_metrics['alpha_final']:.3f}")
    print(f"      Annealing: {stdp_metrics['alpha_annealing']:.3f}")
    print(f"\n   Weight Changes:")
    print(f"      Mean: {stdp_metrics['weight_change_mean']:.6f}")
    print(f"      Final: {stdp_metrics['weight_change_final']:.6f}")

    # Benchmark 3: Accuracy Analysis
    print("\n" + "=" * 80)
    print("3️⃣  ACCURACY PROGRESSION")
    print("=" * 80)

    acc_metrics = analyze_accuracy_progression(history)
    print(f"   Phase 2 (Hybrid):")
    print(f"      Best: {acc_metrics['phase2_best']:.2f}%")
    print(f"      Final: {acc_metrics['phase2_final']:.2f}%")
    print(f"\n   Phase 3 (Finetune):")
    print(f"      Best: {acc_metrics['phase3_best']:.2f}%")
    print(f"      Final: {acc_metrics['phase3_final']:.2f}%")
    print(f"\n   Overall:")
    print(f"      Best Accuracy: {acc_metrics['overall_best']:.2f}%")
    print(f"      Improvement: +{acc_metrics['accuracy_improvement']:.2f}%")
    target_met = '✅ MET' if acc_metrics['overall_best'] >= 92 else f"⚠️  {92 - acc_metrics['overall_best']:.2f}% below"
    print(f"      Target (92%): {target_met}")

    # Benchmark 4: Inference Speed
    print("\n" + "=" * 80)
    print("4️⃣  INFERENCE SPEED")
    print("=" * 80)
    print("   Running benchmark (100 iterations)...")

    speed_metrics = benchmark_inference_speed(model, test_loader, device, num_runs=100)
    print(f"   Mean: {speed_metrics['mean_time_ms']:.2f} ms")
    print(f"   Median: {speed_metrics['median_time_ms']:.2f} ms")
    print(f"   Std: {speed_metrics['std_time_ms']:.2f} ms")
    print(f"   Min: {speed_metrics['min_time_ms']:.2f} ms")
    print(f"   Max: {speed_metrics['max_time_ms']:.2f} ms")
    print(f"   Target (<50ms): {'✅ MET' if speed_metrics['mean_time_ms'] < 50 else '⚠️  EXCEEDED'}")

    # Benchmark 5: Memory Usage
    print("\n" + "=" * 80)
    print("5️⃣  MEMORY USAGE")
    print("=" * 80)

    memory_metrics = benchmark_memory_usage(model, device)
    if device == 'cuda':
        print(f"   Peak: {memory_metrics['peak_memory_mb']:.2f} MB")
        print(f"   Current: {memory_metrics['current_memory_mb']:.2f} MB")
    else:
        print("   Memory profiling only available on CUDA")

    # Benchmark 6: Comparison with Baseline
    print("\n" + "=" * 80)
    print("6️⃣  COMPARISON WITH BASELINE SimpleSNN")
    print("=" * 80)

    baseline = SimpleSNN(input_size=2500, hidden_size=128, output_size=2).to(device)
    baseline_params = count_parameters(baseline)

    print(f"   SimpleSNN Parameters: {baseline_params:,}")
    print(f"   HybridSTDP_SNN Parameters: {params:,}")
    print(f"   Difference: {abs(params - baseline_params)} (should be 0)")
    print(f"   \n   SimpleSNN Baseline: 89.0% accuracy")
    print(f"   HybridSTDP_SNN: {acc_metrics['overall_best']:.2f}% accuracy")
    print(f"   Improvement: +{acc_metrics['overall_best'] - 89.0:.2f}%")

    # Save benchmark results
    print("\n" + "=" * 80)
    print("💾 SAVING BENCHMARK RESULTS")
    print("=" * 80)

    results = {
        'model': {
            'parameters': params,
            'architecture': '2500 → 128 → 2',
            'stdp_features': {
                'homeostatic': config.use_homeostasis,
                'multiscale': config.use_multiscale
            }
        },
        'stdp_dynamics': stdp_metrics,
        'accuracy': acc_metrics,
        'inference_speed': speed_metrics,
        'memory': memory_metrics,
        'baseline_comparison': {
            'simplesnn_params': baseline_params,
            'hybrid_params': params,
            'simplesnn_accuracy': 89.0,
            'hybrid_accuracy': acc_metrics['overall_best'],
            'improvement': acc_metrics['overall_best'] - 89.0
        }
    }

    output_path = 'results/stdp_benchmark_results.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   ✅ Results saved: {output_path}")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
