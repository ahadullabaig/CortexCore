"""
Robustness Testing Module
==========================

Test model performance under noise and signal quality degradation.

Owner: Phase 2 Implementation
Date: 2025-11-09
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Callable
from tqdm import tqdm


def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise at specified Signal-to-Noise Ratio

    SNR (dB) = 10 * log10(signal_power / noise_power)

    Args:
        signal: Input signal
        snr_db: Target SNR in decibels (higher = less noise)

    Returns:
        Noisy signal

    Example:
        >>> clean_signal = np.random.randn(2500)
        >>> noisy_signal = add_gaussian_noise(clean_signal, snr_db=20)
        >>> # Signal with 20dB SNR (moderate noise)
    """
    # Calculate signal power
    signal_power = np.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate required noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return signal + noise


def add_baseline_wander(
    signal: np.ndarray,
    frequency_hz: float = 0.3,
    amplitude: float = 0.2,
    sampling_rate: int = 250
) -> np.ndarray:
    """
    Add baseline wander (low-frequency drift) to simulate breathing artifacts

    Args:
        signal: Input ECG signal
        frequency_hz: Wander frequency (typical: 0.2-0.5 Hz for breathing)
        amplitude: Wander amplitude relative to signal range
        sampling_rate: Signal sampling rate in Hz

    Returns:
        Signal with baseline wander

    Example:
        >>> ecg = np.random.randn(2500)  # 10s at 250Hz
        >>> wandering_ecg = add_baseline_wander(ecg, frequency_hz=0.3, amplitude=0.2)
    """
    n_samples = len(signal)
    t = np.linspace(0, n_samples / sampling_rate, n_samples)

    # Create sinusoidal wander
    wander = amplitude * np.sin(2 * np.pi * frequency_hz * t)

    return signal + wander


def add_motion_artifacts(
    signal: np.ndarray,
    n_spikes: int = 3,
    spike_amplitude: float = 2.0
) -> np.ndarray:
    """
    Add motion artifacts (sudden amplitude spikes) to simulate patient movement

    Args:
        signal: Input ECG signal
        n_spikes: Number of artifact spikes to add
        spike_amplitude: Amplitude of spikes (in signal std units)

    Returns:
        Signal with motion artifacts

    Example:
        >>> ecg = np.random.randn(2500)
        >>> noisy_ecg = add_motion_artifacts(ecg, n_spikes=3, spike_amplitude=2.0)
    """
    noisy_signal = signal.copy()

    # Randomly select spike locations
    spike_indices = np.random.choice(len(signal), n_spikes, replace=False)

    # Add spikes with random polarity
    signal_std = np.std(signal)
    for idx in spike_indices:
        polarity = np.random.choice([-1, 1])
        noisy_signal[idx] += polarity * spike_amplitude * signal_std

    return noisy_signal


def reduce_amplitude(signal: np.ndarray, reduction_factor: float = 0.5) -> np.ndarray:
    """
    Reduce signal amplitude to simulate poor electrode contact

    Args:
        signal: Input signal
        reduction_factor: Amplitude scaling factor (0.5 = 50% amplitude)

    Returns:
        Amplitude-reduced signal

    Example:
        >>> ecg = np.random.randn(2500)
        >>> weak_ecg = reduce_amplitude(ecg, reduction_factor=0.5)
    """
    return signal * reduction_factor


def test_additive_noise_robustness(
    model: torch.nn.Module,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    predict_fn: Callable,
    snr_levels_db: List[float] = [30, 20, 10],
    ensemble_size: int = 3,
    device: str = 'cuda',
    max_samples: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Test model performance under additive Gaussian noise at various SNR levels

    Args:
        model: Trained model
        test_signals: Test signals [n_samples, signal_length]
        test_labels: Test labels [n_samples]
        predict_fn: Prediction function (should accept model, signal, device, ensemble_size)
        snr_levels_db: SNR levels to test (dB)
        ensemble_size: Ensemble size for predictions
        device: Device for inference
        max_samples: Maximum samples to test (None = all)

    Returns:
        Dictionary with results for each SNR level:
        {
            '30dB': {
                'accuracy': 0.87,
                'sensitivity': 0.68,
                'specificity': 0.95,
                'degradation_pct': 2.3,  # vs clean
                'predictions': [...],
                'confidences': [...]
            },
            ...
        }

    Example:
        >>> from src.inference import ensemble_predict
        >>> results = test_additive_noise_robustness(
        ...     model, test_signals, test_labels,
        ...     predict_fn=lambda m, s, d, e: ensemble_predict(m, s, ensemble_size=e, device=d),
        ...     snr_levels_db=[30, 20, 10]
        ... )
    """
    if max_samples is not None:
        test_signals = test_signals[:max_samples]
        test_labels = test_labels[:max_samples]

    results = {}

    # Test clean signals first (baseline)
    print(f"\n🔍 Testing clean signals (baseline)...")
    clean_predictions = []
    clean_confidences = []

    for signal in tqdm(test_signals, desc="Clean signals"):
        result = predict_fn(model, signal, device, ensemble_size)
        clean_predictions.append(result['prediction'])
        clean_confidences.append(result['confidence'])

    clean_predictions = np.array(clean_predictions)
    clean_accuracy = (clean_predictions == test_labels).mean()

    results['clean'] = {
        'accuracy': float(clean_accuracy),
        'predictions': clean_predictions.tolist(),
        'confidences': clean_confidences
    }

    # Test each SNR level
    for snr_db in snr_levels_db:
        print(f"\n🔍 Testing SNR = {snr_db}dB...")

        noisy_predictions = []
        noisy_confidences = []

        for signal in tqdm(test_signals, desc=f"SNR {snr_db}dB"):
            # Add noise
            noisy_signal = add_gaussian_noise(signal, snr_db)

            # Predict
            result = predict_fn(model, noisy_signal, device, ensemble_size)
            noisy_predictions.append(result['prediction'])
            noisy_confidences.append(result['confidence'])

        noisy_predictions = np.array(noisy_predictions)
        noisy_accuracy = (noisy_predictions == test_labels).mean()

        # Calculate per-class metrics
        class_0_mask = test_labels == 0
        class_1_mask = test_labels == 1

        class_0_accuracy = (noisy_predictions[class_0_mask] == test_labels[class_0_mask]).mean() if class_0_mask.any() else 0.0
        class_1_accuracy = (noisy_predictions[class_1_mask] == test_labels[class_1_mask]).mean() if class_1_mask.any() else 0.0

        # Calculate degradation
        degradation_pct = ((clean_accuracy - noisy_accuracy) / clean_accuracy) * 100

        results[f'{snr_db}dB'] = {
            'snr_db': snr_db,
            'accuracy': float(noisy_accuracy),
            'class_0_accuracy': float(class_0_accuracy),
            'class_1_accuracy': float(class_1_accuracy),
            'degradation_pct': float(degradation_pct),
            'degradation_absolute': float(clean_accuracy - noisy_accuracy),
            'predictions': noisy_predictions.tolist(),
            'confidences': noisy_confidences,
            'mean_confidence': float(np.mean(noisy_confidences)),
            'std_confidence': float(np.std(noisy_confidences))
        }

        print(f"   Accuracy: {noisy_accuracy:.1%} (degradation: {degradation_pct:.1f}%)")

    return results


def test_signal_quality_variations(
    model: torch.nn.Module,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    predict_fn: Callable,
    ensemble_size: int = 3,
    device: str = 'cuda',
    max_samples: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Test model under various signal quality degradations

    Args:
        model: Trained model
        test_signals: Test signals
        test_labels: Test labels
        predict_fn: Prediction function
        ensemble_size: Ensemble size
        device: Device
        max_samples: Maximum samples to test

    Returns:
        Dictionary with results for each degradation type:
        {
            'baseline_wander': {...},
            'motion_artifacts': {...},
            'amplitude_reduction': {...},
            'combined': {...}
        }
    """
    if max_samples is not None:
        test_signals = test_signals[:max_samples]
        test_labels = test_labels[:max_samples]

    results = {}

    # Test baseline wander
    print(f"\n🔍 Testing baseline wander...")
    predictions, confidences = _test_degradation(
        model, test_signals, test_labels, predict_fn,
        degradation_fn=lambda s: add_baseline_wander(s, frequency_hz=0.3, amplitude=0.2),
        ensemble_size=ensemble_size, device=device,
        desc="Baseline wander"
    )
    accuracy = (predictions == test_labels).mean()
    results['baseline_wander'] = {
        'accuracy': float(accuracy),
        'predictions': predictions.tolist(),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    print(f"   Accuracy: {accuracy:.1%}")

    # Test motion artifacts
    print(f"\n🔍 Testing motion artifacts...")
    predictions, confidences = _test_degradation(
        model, test_signals, test_labels, predict_fn,
        degradation_fn=lambda s: add_motion_artifacts(s, n_spikes=3, spike_amplitude=2.0),
        ensemble_size=ensemble_size, device=device,
        desc="Motion artifacts"
    )
    accuracy = (predictions == test_labels).mean()
    results['motion_artifacts'] = {
        'accuracy': float(accuracy),
        'predictions': predictions.tolist(),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    print(f"   Accuracy: {accuracy:.1%}")

    # Test amplitude reduction
    print(f"\n🔍 Testing amplitude reduction...")
    predictions, confidences = _test_degradation(
        model, test_signals, test_labels, predict_fn,
        degradation_fn=lambda s: reduce_amplitude(s, reduction_factor=0.5),
        ensemble_size=ensemble_size, device=device,
        desc="Amplitude reduction"
    )
    accuracy = (predictions == test_labels).mean()
    results['amplitude_reduction'] = {
        'accuracy': float(accuracy),
        'predictions': predictions.tolist(),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    print(f"   Accuracy: {accuracy:.1%}")

    # Test combined degradations
    print(f"\n🔍 Testing combined degradations...")
    def combined_degradation(signal):
        s = add_baseline_wander(signal, frequency_hz=0.3, amplitude=0.15)
        s = add_motion_artifacts(s, n_spikes=2, spike_amplitude=1.5)
        s = reduce_amplitude(s, reduction_factor=0.7)
        return s

    predictions, confidences = _test_degradation(
        model, test_signals, test_labels, predict_fn,
        degradation_fn=combined_degradation,
        ensemble_size=ensemble_size, device=device,
        desc="Combined degradations"
    )
    accuracy = (predictions == test_labels).mean()
    results['combined'] = {
        'accuracy': float(accuracy),
        'predictions': predictions.tolist(),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    print(f"   Accuracy: {accuracy:.1%}")

    return results


def _test_degradation(
    model: torch.nn.Module,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    predict_fn: Callable,
    degradation_fn: Callable,
    ensemble_size: int,
    device: str,
    desc: str
) -> Tuple[np.ndarray, List[float]]:
    """
    Helper function to test a specific degradation

    Args:
        model: Model
        test_signals: Test signals
        test_labels: Test labels
        predict_fn: Prediction function
        degradation_fn: Function to degrade signal
        ensemble_size: Ensemble size
        device: Device
        desc: Description for progress bar

    Returns:
        Tuple of (predictions, confidences)
    """
    predictions = []
    confidences = []

    for signal in tqdm(test_signals, desc=desc):
        # Apply degradation
        degraded_signal = degradation_fn(signal)

        # Predict
        result = predict_fn(model, degraded_signal, device, ensemble_size)
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])

    return np.array(predictions), confidences
