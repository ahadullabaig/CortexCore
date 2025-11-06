#!/usr/bin/env python3
"""Test the spike visualization endpoint"""

import requests
import json

# Generate a sample first
print("📊 Generating test ECG sample...")
gen_response = requests.post(
    'http://localhost:5000/api/generate_sample',
    json={'condition': 'normal'}
)

if gen_response.status_code == 200:
    sample_data = gen_response.json()
    signal = sample_data['signal']
    print(f"✅ Generated {len(signal)} samples")

    # Test spike visualization
    print("\n🎨 Testing spike visualization...")
    viz_response = requests.post(
        'http://localhost:5000/api/visualize_spikes',
        json={'signal': signal, 'num_steps': 100, 'gain': 10.0}
    )

    if viz_response.status_code == 200:
        result = viz_response.json()
        print("\n✅ VISUALIZATION SUCCESSFUL!")
        print(f"   Spike times recorded: {len(result.get('spike_times', []))}")
        print(f"   Neuron IDs recorded: {len(result.get('neuron_ids', []))}")
        print(f"   Time steps: {result.get('num_steps')}")
        print(f"   Neurons visualized: {result.get('num_neurons')}")
        print(f"   Total spikes: {result.get('total_spikes')}")
        print(f"   Spike rate: {result.get('spike_rate', 0):.4f}")
        print(f"   Gain: {result.get('gain')}")
    else:
        print(f"\n❌ Visualization failed: {viz_response.status_code}")
        print(f"   Error: {viz_response.text}")
else:
    print(f"❌ Sample generation failed: {gen_response.status_code}")
