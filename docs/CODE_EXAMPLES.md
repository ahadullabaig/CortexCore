# Code Examples

Quick reference for common coding patterns in the neuromorphic SNN project.

## Loading and Running a Trained SNN

```python
from src.model import SimpleSNN
from src.inference import load_model, predict
from src.utils import get_device

device = get_device()
model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
load_model(model, 'models/best_model.pt', device)

# Predict on new data (expects spike-encoded input)
prediction = predict(model, spike_encoded_signal, device)
```

## Training a Custom SNN

```python
from src.model import SimpleSNN
from src.train import train_model
from src.data import ECGDataset
from torch.utils.data import DataLoader

# Load data
train_dataset = ECGDataset('data/synthetic/train_ecg.pt')
val_dataset = ECGDataset('data/synthetic/val_ecg.pt')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Create model and train
model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
train_model(model, train_loader, val_loader, num_epochs=50, device='cuda')
```

## Generating and Encoding ECG Data

```python
from src.data import generate_synthetic_ecg, rate_encode
from src.utils import set_seed

# Generate synthetic ECG
set_seed(42)  # For reproducibility
ecg_signal = generate_synthetic_ecg(
    duration=10,           # seconds
    sampling_rate=250,     # Hz
    heart_rate=70,         # BPM
    condition='normal'     # or 'arrhythmia'
)

# Convert to spike train
spike_train = rate_encode(
    ecg_signal,
    num_steps=100,   # time steps
    gain=10.0        # spike rate scaling
)
# Output shape: [100, 2500] = [time_steps, features]
```

## Creating a Custom SNN Architecture

```python
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class CustomSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # Ensure time-first: [time_steps, batch, features]
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)

        # Initialize states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []
        mem_rec = []

        # Process each time step
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)
```

## Calculating Energy Efficiency

```python
from src.model import measure_energy_efficiency

# After forward pass
spikes, membrane = model(spike_encoded_input)
metrics = measure_energy_efficiency(spikes)

print(f"Spike sparsity: {metrics['sparsity']:.2%}")
print(f"Operation reduction: {metrics['ops_reduction']:.2%}")
print(f"Total spikes: {metrics['total_spikes']}")
```

## Quick STDP Example

For full STDP implementation, see [STDP_GUIDE.md](./STDP_GUIDE.md).

```python
import torch
import torch.nn as nn
import snntorch as snn

class QuickSTDP_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2500, 128)
        self.lif1 = snn.Leaky(beta=0.9)

        # STDP parameters
        self.a_plus = 0.01   # LTP learning rate
        self.a_minus = 0.01  # LTD learning rate
        self.tau = 20.0      # Time constant

    def stdp_update(self, pre_spikes, post_spikes, weight):
        """Simple STDP weight update"""
        delta_w = torch.zeros_like(weight)

        for t in range(pre_spikes.size(0)):
            # Compute spike correlations
            correlation = torch.einsum('bi,bj->ij',
                                       pre_spikes[t],
                                       post_spikes[t])
            delta_w += self.a_plus * correlation / pre_spikes.size(1)

        return torch.clamp(weight + delta_w, 0.0, 1.0)

# Hybrid training: STDP + Backprop
model = QuickSTDP_SNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for data, targets in train_loader:
        # Forward pass
        spk1, spk2 = model(data)

        # STDP for layer 1 (unsupervised)
        with torch.no_grad():
            model.fc1.weight.data = model.stdp_update(
                data, spk1, model.fc1.weight.data
            )

        # Backprop for output layer (supervised)
        optimizer.zero_grad()
        output = spk2.sum(dim=0)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
```

## Visualizing STDP Weights

```python
import matplotlib.pyplot as plt

def plot_stdp_weights(model):
    weights = model.fc1.weight.data.cpu().numpy()
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('STDP-Learned Synaptic Weights')
    plt.xlabel('Post-synaptic Neurons')
    plt.ylabel('Pre-synaptic Features')
    plt.savefig('results/plots/stdp_weights.png')
    return weights
```
