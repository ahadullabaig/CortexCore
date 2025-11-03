"""
Test Inference Module
=====================

Tests the inference functionality with the trained model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.model import SimpleSNN
from src.inference import load_model, predict, profile_inference
from src.utils import get_device

def main():
    print("="*60)
    print("🧪 Testing Inference Module")
    print("="*60)

    device = get_device()
    print(f"\n📱 Device: {device}")

    # Check if trained model exists
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train the model first: python scripts/train_snn_mvp.py")
        return

    # Load test data
    print(f"\n📂 Loading test data...")
    test_data = torch.load("data/synthetic/test_data.pt")
    print(f"   Test samples: {test_data['signals'].shape[0]}")

    # Create and load model
    print(f"\n🏗️  Loading trained model...")
    model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
    model = load_model(str(model_path), model, device=str(device))
    print(f"   ✅ Model loaded successfully")

    # Test single prediction
    print(f"\n🔮 Testing single prediction...")
    test_signal = test_data['signals'][0].numpy()
    test_label = test_data['labels'][0].item()

    result = predict(model, test_signal, device=str(device), num_steps=100)

    print(f"   True label: {test_label} ({'Normal' if test_label == 0 else 'Arrhythmia'})")
    print(f"   Prediction: {result['prediction']} ({result['class_name']})")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Inference time: {result['inference_time_ms']:.2f} ms")
    print(f"   Spike count: {result['spike_count']:.0f}")
    print(f"   Probabilities: {result['probabilities']}")

    correct = result['prediction'] == test_label
    print(f"   Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")

    # Test batch prediction
    print(f"\n📊 Testing batch predictions (first 10 samples)...")
    correct_count = 0
    total_time = 0

    for i in range(min(10, len(test_data['signals']))):
        signal = test_data['signals'][i].numpy()
        label = test_data['labels'][i].item()

        result = predict(model, signal, device=str(device), num_steps=100, return_confidence=False)

        if result['prediction'] == label:
            correct_count += 1

        total_time += result['inference_time_ms']

    accuracy = 100.0 * correct_count / 10
    avg_time = total_time / 10

    print(f"   Accuracy: {accuracy:.1f}% ({correct_count}/10)")
    print(f"   Avg inference time: {avg_time:.2f} ms")
    print(f"   MVP target (<100ms): {'✅ PASS' if avg_time < 100 else '❌ FAIL'}")

    # Profile inference
    print(f"\n⚡ Profiling inference performance...")
    profile_metrics = profile_inference(
        model,
        input_shape=(100, 1, 2500),  # [time_steps, batch, features]
        n_iterations=50,
        device=str(device)
    )

    print(f"   Mean time: {profile_metrics['mean_time_ms']:.2f} ms")
    print(f"   Std time: {profile_metrics['std_time_ms']:.2f} ms")
    print(f"   Min time: {profile_metrics['min_time_ms']:.2f} ms")
    print(f"   Max time: {profile_metrics['max_time_ms']:.2f} ms")
    print(f"   Throughput: {profile_metrics['throughput_samples_per_sec']:.1f} samples/sec")

    # Summary
    print("\n" + "="*60)
    print("✅ Inference Module Tests Complete!")
    print("="*60)

    if avg_time < 100:
        print("🎉 MVP inference time target achieved (<100ms)")
    else:
        print(f"⚠️  Inference time {avg_time:.2f}ms exceeds MVP target of 100ms")

if __name__ == "__main__":
    main()
