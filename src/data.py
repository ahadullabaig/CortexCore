"""
Data Generation & Preprocessing Module
======================================

Owner: CS3 / Data Engineer

Responsibilities:
- Synthetic ECG/EEG generation
- Signal preprocessing
- Spike encoding
- Data loading and caching

Phase: Days 1-30
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import neurokit2 as nk

# ============================================
# TODO: Day 1-2 - Basic Data Generation
# ============================================

def generate_synthetic_ecg(
    n_samples: int = 100,
    duration: int = 10,
    sampling_rate: int = 250,
    condition: str = 'normal'
) -> np.ndarray:
    """
    Generate synthetic ECG signals

    Args:
        n_samples: Number of samples to generate
        duration: Duration in seconds
        sampling_rate: Sampling frequency in Hz
        condition: 'normal' or 'arrhythmia'

    Returns:
        ECG signals: shape [n_samples, duration * sampling_rate]

    TODO:
        - Add more pathology types
        - Implement patient variability
        - Add realistic noise models
    """
    signals = []

    for i in range(n_samples):
        if condition == 'normal':
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=np.random.randint(60, 80),
                noise=0.05
            )
        elif condition == 'arrhythmia':
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=np.random.randint(100, 140),
                noise=0.1
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Normalize
        ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
        signals.append(ecg)

    return np.array(signals)


# ============================================
# TODO: Day 2-3 - Spike Encoding
# ============================================

def rate_encode(
    signal: np.ndarray,
    num_steps: int = 100,
    gain: float = 10.0
) -> np.ndarray:
    """
    Convert continuous signal to spike train using rate coding

    Args:
        signal: Input signal
        num_steps: Number of time steps
        gain: Spike rate multiplier

    Returns:
        Spike train: shape [num_steps]

    TODO:
        - Implement temporal coding
        - Add latency coding
        - Try population coding
    """
    # Normalize to [0, 1]
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

    # Bin signal into time steps
    bins = np.linspace(0, len(signal), num_steps + 1, dtype=int)
    rates = np.array([signal_norm[bins[i]:bins[i+1]].mean() for i in range(num_steps)])

    # Generate spikes (Poisson process)
    spikes = np.random.rand(num_steps) < (rates * gain)

    return spikes.astype(np.float32)


# ============================================
# TODO: Day 3-4 - Dataset Class
# ============================================

class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals

    TODO:
        - Add data augmentation
        - Implement caching
        - Add multi-modal support (ECG + EEG)
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, encode_spikes: bool = True):
        """
        Args:
            signals: ECG signals [n_samples, signal_length]
            labels: Class labels [n_samples]
            encode_spikes: Whether to encode as spikes
        """
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.encode_spikes = encode_spikes

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.encode_spikes:
            # TODO: Implement proper spike encoding here
            signal = signal  # Placeholder

        return signal, label


def load_dataset(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Load dataset and create DataLoader

    Args:
        data_path: Path to .pt file
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance

    TODO:
        - Add validation split
        - Implement stratified sampling
    """
    data = torch.load(data_path)

    dataset = ECGDataset(
        signals=data['signals'].numpy() if isinstance(data['signals'], torch.Tensor) else data['signals'],
        labels=data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================
# TODO: Day 5-7 - Preprocessing
# ============================================

def preprocess_signal(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Preprocess ECG signal

    Args:
        signal: Raw ECG signal
        fs: Sampling frequency

    Returns:
        Preprocessed signal

    TODO:
        - Bandpass filtering
        - Baseline drift correction
        - Artifact removal
        - Peak detection
    """
    # Placeholder - implement filtering
    return signal


# ============================================
# TODO: Day 8-14 - Advanced Features
# ============================================

def generate_eeg_data():
    """TODO: Implement EEG generation"""
    raise NotImplementedError("EEG generation - Week 2 task")


def augment_data():
    """TODO: Implement data augmentation"""
    raise NotImplementedError("Data augmentation - Week 2 task")


def create_real_time_pipeline():
    """TODO: Implement streaming pipeline"""
    raise NotImplementedError("Real-time pipeline - Week 3 task")


# ============================================
# Utility Functions
# ============================================

def save_dataset(data: dict, path: str):
    """Save dataset to disk"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"✅ Dataset saved to {path}")


def dataset_statistics(signals: np.ndarray, labels: np.ndarray) -> dict:
    """Calculate dataset statistics"""
    stats = {
        'n_samples': len(signals),
        'signal_shape': signals.shape,
        'n_classes': len(np.unique(labels)),
        'class_distribution': {
            int(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        'signal_mean': float(np.mean(signals)),
        'signal_std': float(np.std(signals)),
    }
    return stats


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 Testing data module...")

    # Generate sample data
    print("\n1. Generating ECG samples...")
    normal_ecg = generate_synthetic_ecg(n_samples=10, condition='normal')
    arrhythmia_ecg = generate_synthetic_ecg(n_samples=10, condition='arrhythmia')
    print(f"   Normal ECG: {normal_ecg.shape}")
    print(f"   Arrhythmia ECG: {arrhythmia_ecg.shape}")

    # Test spike encoding
    print("\n2. Testing spike encoding...")
    spikes = rate_encode(normal_ecg[0])
    print(f"   Spikes: {spikes.sum()}/{len(spikes)} ({spikes.sum()/len(spikes)*100:.1f}% active)")

    # Create dataset
    print("\n3. Creating dataset...")
    all_signals = np.concatenate([normal_ecg, arrhythmia_ecg], axis=0)
    all_labels = np.array([0]*10 + [1]*10)

    dataset = ECGDataset(all_signals, all_labels)
    print(f"   Dataset size: {len(dataset)}")

    # Test dataloader
    print("\n4. Testing dataloader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_signals, batch_labels = next(iter(loader))
    print(f"   Batch shape: {batch_signals.shape}")
    print(f"   Labels: {batch_labels}")

    print("\n✅ Data module working!")
