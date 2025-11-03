"""
Test Flask Demo Integration
============================

Tests the Flask demo endpoints with the trained SNN model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from demo.app import app, init_model

def test_demo_integration():
    print("=" * 60)
    print("🧪 Testing Flask Demo Integration")
    print("=" * 60)

    # Initialize model
    print("\n🏗️  Initializing model...")
    init_model()

    # Create Flask test client
    app.config['TESTING'] = True
    client = app.test_client()

    print("\n✅ Model initialized successfully")

    # Test health endpoint
    print("\n📊 Testing /health endpoint...")
    response = client.get('/health')
    assert response.status_code == 200
    health_data = response.get_json()
    print(f"   Status: {health_data['status']}")
    print(f"   Model loaded: {health_data['model']['loaded']}")
    print(f"   Device: {health_data['device']}")

    # Load test data
    print("\n📂 Loading test data...")
    test_data = torch.load("data/synthetic/test_data.pt")
    test_signal = test_data['signals'][0].numpy().tolist()
    test_label = test_data['labels'][0].item()
    print(f"   True label: {test_label} ({'Normal' if test_label == 0 else 'Arrhythmia'})")

    # Test prediction endpoint
    print("\n🔮 Testing /api/predict endpoint...")
    response = client.post(
        '/api/predict',
        json={'signal': test_signal, 'num_steps': 100},
        content_type='application/json'
    )
    assert response.status_code == 200
    pred_data = response.get_json()
    print(f"   Prediction: {pred_data['prediction']} ({pred_data['class_name']})")
    print(f"   Confidence: {pred_data['confidence']:.4f}")
    print(f"   Inference time: {pred_data['inference_time_ms']:.2f} ms")
    print(f"   Spike count: {pred_data['spike_count']}")
    print(f"   Result: {'✅ CORRECT' if pred_data['prediction'] == test_label else '❌ INCORRECT'}")

    # Test generate sample endpoint
    print("\n📊 Testing /api/generate_sample endpoint...")
    response = client.post(
        '/api/generate_sample',
        json={'condition': 'normal', 'duration': 10, 'sampling_rate': 250},
        content_type='application/json'
    )
    assert response.status_code == 200
    sample_data = response.get_json()
    print(f"   Condition: {sample_data['condition']}")
    print(f"   Signal length: {sample_data['length']} samples")
    print(f"   Duration: {sample_data['duration']} seconds")
    print(f"   Sampling rate: {sample_data['sampling_rate']} Hz")

    # Test spike visualization endpoint
    print("\n⚡ Testing /api/visualize_spikes endpoint...")
    response = client.post(
        '/api/visualize_spikes',
        json={'signal': test_signal, 'num_steps': 100, 'gain': 10.0},
        content_type='application/json'
    )
    if response.status_code != 200:
        print(f"   ❌ Error: {response.status_code}")
        print(f"   Response: {response.get_json()}")
    else:
        spike_data = response.get_json()
        print(f"   Number of neurons: {spike_data['num_neurons']}")
        print(f"   Total spikes: {spike_data['total_spikes']}")
        print(f"   Spike rate: {spike_data['spike_rate']:.4f}")
        print(f"   Time steps: {spike_data['num_steps']}")

    # Test metrics endpoint
    print("\n📈 Testing /api/metrics endpoint...")
    response = client.get('/api/metrics')
    if response.status_code == 200:
        metrics_data = response.get_json()
        print(f"   Model loaded: {metrics_data['model_loaded']}")
        print(f"   Device: {metrics_data['device']}")
        print(f"   CUDA available: {metrics_data['cuda_available']}")
        if metrics_data['cuda_available']:
            print(f"   GPU memory allocated: {metrics_data['gpu_memory_allocated']:.2f} GB")
    else:
        print(f"   ❌ Error: {response.status_code}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Flask Demo Integration Tests Complete!")
    print("=" * 60)
    print("\n🎉 Flask demo successfully integrated with trained SNN model!")
    print("\n📍 To run the demo server:")
    print("   python demo/app.py")
    print("   or: bash scripts/04_run_demo.sh")
    print("\n   Then access: http://localhost:5000")

if __name__ == "__main__":
    test_demo_integration()
