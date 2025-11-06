#!/usr/bin/env python3
"""Test predictions on both Normal and Arrhythmia samples"""

import requests

def test_condition(condition):
    """Test prediction for a specific condition"""
    print(f"\n{'='*60}")
    print(f"Testing: {condition.upper()}")
    print('='*60)

    # Generate sample
    gen_response = requests.post(
        'http://localhost:5000/api/generate_sample',
        json={'condition': condition}
    )

    if gen_response.status_code != 200:
        print(f"❌ Failed to generate {condition} sample")
        return

    signal = gen_response.json()['signal']
    print(f"✅ Generated {len(signal)} samples for {condition}")

    # Predict
    pred_response = requests.post(
        'http://localhost:5000/api/predict',
        json={'signal': signal, 'num_steps': 100}
    )

    if pred_response.status_code != 200:
        print(f"❌ Prediction failed: {pred_response.text}")
        return

    result = pred_response.json()
    print(f"\n🧠 STDP Model Prediction:")
    print(f"   Input: {condition}")
    print(f"   Predicted: {result.get('class_name')}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")
    print(f"   Probabilities:")
    probs = result.get('probabilities', [])
    print(f"     - Normal:     {probs[0]:.6f}")
    print(f"     - Arrhythmia: {probs[1]:.6f}")
    print(f"   Inference time: {result.get('inference_time_ms', 0):.2f} ms")
    print(f"   Spike count: {result.get('spike_count')}")

    # Check if prediction matches input
    expected = 'Normal' if condition == 'normal' else 'Arrhythmia'
    actual = result.get('class_name')
    if expected == actual:
        print(f"   ✅ Correct prediction!")
    else:
        print(f"   ⚠️  Predicted {actual}, expected {expected}")

print("="*60)
print("🧠 CortexCore STDP Model - Demo Test")
print("="*60)
print("\nModel: HybridSTDP_SNN (90.3% accuracy)")
print("Features: Multi-timescale STDP + Homeostatic plasticity")
print("Parameters: 320,386")

# Test both conditions
test_condition('normal')
test_condition('arrhythmia')

print("\n" + "="*60)
print("✅ Demo testing complete!")
print("="*60)
