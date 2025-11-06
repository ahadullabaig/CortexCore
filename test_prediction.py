#!/usr/bin/env python3
"""Test the STDP model prediction endpoint"""

import requests
import json

# Generate a sample first
print("📊 Generating test ECG sample...")
gen_response = requests.post(
    'http://localhost:5000/api/generate_sample',
    json={'condition': 'normal', 'duration': 10, 'sampling_rate': 250}
)

if gen_response.status_code == 200:
    sample_data = gen_response.json()
    signal = sample_data['signal']
    print(f"✅ Generated {len(signal)} samples")

    # Test prediction
    print("\n🧠 Testing prediction with STDP model...")
    pred_response = requests.post(
        'http://localhost:5000/api/predict',
        json={'signal': signal, 'num_steps': 100}
    )

    if pred_response.status_code == 200:
        result = pred_response.json()
        print("\n✅ PREDICTION SUCCESSFUL!")
        print(f"   Prediction: {result.get('prediction')}")
        print(f"   Class: {result.get('class_name')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Probabilities: {result.get('probabilities')}")
        print(f"   Inference time: {result.get('inference_time_ms', 0):.2f} ms")
        print(f"   Spike count: {result.get('spike_count')}")
    else:
        print(f"\n❌ Prediction failed: {pred_response.status_code}")
        print(f"   Error: {pred_response.text}")
else:
    print(f"❌ Sample generation failed: {gen_response.status_code}")
    print(f"   Error: {gen_response.text}")
