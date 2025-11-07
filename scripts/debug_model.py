"""
Debug script to understand model prediction behavior
"""
import torch
import numpy as np
from src.model import SimpleSNN
from src.data import generate_synthetic_ecg
from src.inference import predict
from src.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Load model
print("=" * 60)
print("MODEL DIAGNOSIS")
print("=" * 60)

checkpoint = torch.load('models/best_model.pt', map_location='cpu')
config = checkpoint.get('config', {})

model = SimpleSNN(
    input_size=config.get('input_size', 2500),
    hidden_size=config.get('hidden_size', 128),
    output_size=2
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✅ Model loaded: {checkpoint.get('epoch')} epochs, {checkpoint.get('val_acc')}% val acc")

# Generate test signals
print("\n" + "=" * 60)
print("TESTING WITH SYNTHETIC SIGNALS")
print("=" * 60)

# Test Normal signals
print("\n📊 Testing Normal ECG signals:")
normal_signals = generate_synthetic_ecg(n_samples=5, condition='normal')
for i, signal in enumerate(normal_signals):
    result = predict(model, signal, device='cpu', num_steps=100, gain=10.0)
    print(f"  Sample {i+1}: {result['class_name']:12s} | Confidence: {result['confidence']:.4f} | Probs: {result['probabilities']}")

# Test Arrhythmia signals
print("\n📊 Testing Arrhythmia ECG signals:")
arrhythmia_signals = generate_synthetic_ecg(n_samples=5, condition='arrhythmia')
for i, signal in enumerate(arrhythmia_signals):
    result = predict(model, signal, device='cpu', num_steps=100, gain=10.0)
    print(f"  Sample {i+1}: {result['class_name']:12s} | Confidence: {result['confidence']:.4f} | Probs: {result['probabilities']}")

# Test with raw model output to see logits
print("\n" + "=" * 60)
print("RAW MODEL OUTPUT (LOGITS)")
print("=" * 60)

print("\n📊 Normal signal logits:")
signal = normal_signals[0]
signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
spikes = np.random.rand(100, len(signal)) < (signal_norm * 10.0 / 100.0)
input_data = torch.FloatTensor(spikes).unsqueeze(1)  # [100, 1, 2500]

with torch.no_grad():
    output = model(input_data)
    if isinstance(output, tuple):
        spikes_out, membrane = output
        logits = spikes_out.sum(dim=0)  # [1, 2]
    else:
        logits = output

    print(f"  Logits: {logits[0].numpy()}")
    print(f"  Softmax: {torch.softmax(logits, dim=1)[0].numpy()}")
    print(f"  Prediction: {logits.argmax(dim=1).item()}")

print("\n📊 Arrhythmia signal logits:")
signal = arrhythmia_signals[0]
signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
spikes = np.random.rand(100, len(signal)) < (signal_norm * 10.0 / 100.0)
input_data = torch.FloatTensor(spikes).unsqueeze(1)

with torch.no_grad():
    output = model(input_data)
    if isinstance(output, tuple):
        spikes_out, membrane = output
        logits = spikes_out.sum(dim=0)
    else:
        logits = output

    print(f"  Logits: {logits[0].numpy()}")
    print(f"  Softmax: {torch.softmax(logits, dim=1)[0].numpy()}")
    print(f"  Prediction: {logits.argmax(dim=1).item()}")

# Check spike activity
print("\n" + "=" * 60)
print("SPIKE ACTIVITY ANALYSIS")
print("=" * 60)

print("\n📊 Normal signal spike activity:")
signal = normal_signals[0]
signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
spikes = np.random.rand(100, len(signal)) < (signal_norm * 10.0 / 100.0)
input_data = torch.FloatTensor(spikes).unsqueeze(1)

with torch.no_grad():
    output = model(input_data)
    if isinstance(output, tuple):
        spikes_out, membrane = output
        print(f"  Input spikes: {spikes.sum()}")
        print(f"  Output spike count: {spikes_out.sum().item()}")
        print(f"  Output spike rate: {spikes_out.mean().item():.4f}")
        print(f"  Spikes per class: {spikes_out.sum(dim=0)[0].numpy()}")

print("\n📊 Arrhythmia signal spike activity:")
signal = arrhythmia_signals[0]
signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
spikes = np.random.rand(100, len(signal)) < (signal_norm * 10.0 / 100.0)
input_data = torch.FloatTensor(spikes).unsqueeze(1)

with torch.no_grad():
    output = model(input_data)
    if isinstance(output, tuple):
        spikes_out, membrane = output
        print(f"  Input spikes: {spikes.sum()}")
        print(f"  Output spike count: {spikes_out.sum().item()}")
        print(f"  Output spike rate: {spikes_out.mean().item():.4f}")
        print(f"  Spikes per class: {spikes_out.sum(dim=0)[0].numpy()}")

print("\n" + "=" * 60)
