"""
MIT-BIH Arrhythmia Database Preprocessing Functions

This module provides reusable preprocessing functions for transforming
MIT-BIH raw ECG data into model-ready format.

Functions:
    - load_raw_record: Load MIT-BIH record with signal and annotations
    - resample_signal: Resample from 360 Hz to 250 Hz
    - filter_signal: Apply bandpass + notch filtering
    - calculate_sqi: Calculate Signal Quality Index
    - segment_signal: Split signal into 10-second windows
    - map_beat_to_segment: Map beat-level annotations to segment labels
    - normalize_signal: Normalize signal (Z-score, minmax, or robust)
    - augment_segment: Data augmentation (optional)
"""

import numpy as np
import wfdb
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_record(
    record_id: str,
    data_dir: Path = Path('data/mitbih')
) -> Tuple[np.ndarray, wfdb.Annotation, wfdb.Record]:
    """
    Load MIT-BIH record with signal and annotations

    Args:
        record_id: Patient record ID (e.g., '100', '101')
        data_dir: Directory containing MIT-BIH files

    Returns:
        signal: Physical signal [n_samples, 2] in mV (two leads)
        annotation: wfdb.Annotation object with beat locations and symbols
        record: wfdb.Record object with metadata

    Raises:
        FileNotFoundError: If record files not found
        wfdb.io.ReadRecordException: If files are corrupted
    """
    record_path = str(data_dir / record_id)

    try:
        # Load signal (shape: [650000, 2] for 30 min @ 360 Hz)
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # Physical units (mV)

        # Load annotations (.atr file)
        annotation = wfdb.rdann(record_path, 'atr')

        # Validate
        assert record.fs == 360, f"Expected 360 Hz, got {record.fs}"
        assert record.n_sig == 2, f"Expected 2 leads, got {record.n_sig}"
        assert len(annotation.sample) > 0, "No annotations found"

        return signal, annotation, record

    except wfdb.io.ReadRecordException as e:
        logger.error(f"Corrupted file for record {record_id}: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Files not found for record {record_id}: {e}")
        raise


def resample_signal(
    signal: np.ndarray,
    original_fs: int = 360,
    target_fs: int = 250
) -> np.ndarray:
    """
    Resample ECG signal using Fourier method

    Uses scipy.signal.resample which preserves morphology better than
    linear interpolation. Anti-aliasing is built-in.

    Args:
        signal: Input signal [n_samples] or [n_samples, n_channels]
        original_fs: Original sampling rate (360 Hz for MIT-BIH)
        target_fs: Target sampling rate (250 Hz for our model)

    Returns:
        Resampled signal [n_samples_new] or [n_samples_new, n_channels]

    Example:
        30 min @ 360 Hz = 650,000 samples
        30 min @ 250 Hz = 450,000 samples
        Compression ratio: 250/360 = 0.694
    """
    # Calculate new length
    new_length = int(len(signal) * target_fs / original_fs)

    # Resample using Fourier method (handles multi-channel automatically)
    signal_resampled = sp_signal.resample(signal, new_length, axis=0)

    return signal_resampled


def apply_highpass_filter(
    signal: np.ndarray,
    cutoff: float = 0.5,
    fs: int = 250,
    order: int = 4
) -> np.ndarray:
    """
    Remove baseline wander using Butterworth high-pass filter

    Args:
        signal: Input signal
        cutoff: High-pass cutoff frequency (0.5 Hz for monitoring)
        fs: Sampling frequency
        order: Filter order (4 is standard for ECG)

    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Use filtfilt for zero-phase filtering
    signal_filtered = filtfilt(b, a, signal, axis=0)

    return signal_filtered


def apply_lowpass_filter(
    signal: np.ndarray,
    cutoff: float = 40,
    fs: int = 250,
    order: int = 4
) -> np.ndarray:
    """
    Remove high-frequency noise using Butterworth low-pass filter

    Args:
        signal: Input signal
        cutoff: Low-pass cutoff frequency (40 Hz preserves QRS complex)
        fs: Sampling frequency
        order: Filter order

    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    signal_filtered = filtfilt(b, a, signal, axis=0)
    return signal_filtered


def apply_notch_filter(
    signal: np.ndarray,
    notch_freq: float = 60,
    fs: int = 250,
    Q: float = 30
) -> np.ndarray:
    """
    Remove powerline interference using notch filter

    Args:
        signal: Input signal
        notch_freq: Notch frequency (60 Hz for US, 50 Hz for Europe)
        fs: Sampling frequency
        Q: Quality factor (30 = narrow notch)

    Returns:
        Filtered signal
    """
    b, a = iirnotch(notch_freq, Q, fs)
    signal_filtered = filtfilt(b, a, signal, axis=0)
    return signal_filtered


def filter_signal(
    signal: np.ndarray,
    fs: int = 250,
    highpass_cutoff: float = 0.5,
    lowpass_cutoff: float = 40.0,
    notch_freq: float = 60.0,
    filter_order: int = 4,
    notch_q: float = 30.0
) -> np.ndarray:
    """
    Apply complete filter stack: High-pass → Low-pass → Notch

    Args:
        signal: Input signal
        fs: Sampling frequency
        highpass_cutoff: High-pass cutoff frequency
        lowpass_cutoff: Low-pass cutoff frequency
        notch_freq: Notch frequency
        filter_order: Filter order for Butterworth filters
        notch_q: Quality factor for notch filter

    Returns:
        Filtered signal
    """
    signal_filtered = apply_highpass_filter(signal, cutoff=highpass_cutoff, fs=fs, order=filter_order)
    signal_filtered = apply_lowpass_filter(signal_filtered, cutoff=lowpass_cutoff, fs=fs, order=filter_order)
    signal_filtered = apply_notch_filter(signal_filtered, notch_freq=notch_freq, fs=fs, Q=notch_q)
    return signal_filtered


def calculate_sqi(signal: np.ndarray, fs: int = 250) -> float:
    """
    Calculate Signal Quality Index using composite metrics

    Combines multiple quality metrics:
    - Skewness check: |skewness| < 0.8
    - Kurtosis check: kurtosis > 5
    - SNR check: SNR > 10 dB
    - Flatline check: std > 0.05 mV
    - Saturation check: no clipping at ±5 mV

    Args:
        signal: Input signal
        fs: Sampling frequency

    Returns:
        SQI score in [0, 1], where >0.7 is acceptable quality
    """
    # Skewness score (normalized)
    signal_skew = skew(signal)
    skew_score = 1.0 if abs(signal_skew) < 0.8 else 0.0

    # Kurtosis score (normalized)
    signal_kurtosis = kurtosis(signal)
    kurt_score = 1.0 if signal_kurtosis > 5 else max(0.0, signal_kurtosis / 5.0)

    # SNR score (simple noise estimation)
    # Assume high-frequency (>40 Hz) is noise
    from scipy.signal import welch
    f, psd = welch(signal, fs=fs, nperseg=min(fs*2, len(signal)))
    signal_power = psd[(f >= 5) & (f <= 30)].sum()  # QRS band
    noise_power = psd[f > 40].sum()  # High-freq noise
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
    snr_score = min(1.0, max(0.0, snr_db / 20))  # Normalize to [0, 1]

    # Flatline check
    signal_std = np.std(signal)
    flatline_score = 1.0 if signal_std > 0.05 else 0.0

    # Saturation check (clip detection)
    saturation_score = 1.0 if (np.max(signal) < 4.5 and np.min(signal) > -4.5) else 0.0

    # Composite SQI (weighted average)
    sqi = (skew_score * 0.2 + kurt_score * 0.2 + snr_score * 0.3 +
           flatline_score * 0.15 + saturation_score * 0.15)

    return sqi


def map_beat_to_segment(beats: List[str]) -> Tuple[int, float]:
    """
    Map beat-level annotations to segment-level binary label

    Strategy: Conservative labeling (≥1 arrhythmic beat → Arrhythmia segment)

    AAMI EC57 Beat Classification:
        Normal (N-class): N, L, R
        Arrhythmia (A-class): V, A, /, F, f, J, S, E, a, e, j

    Args:
        beats: List of beat symbols from MIT-BIH annotation

    Returns:
        label: 0 (Normal) or 1 (Arrhythmia)
        confidence: Fraction of arrhythmic beats in segment (0.0-1.0)
    """
    # Define arrhythmia beat types (AAMI EC57 + paced)
    arrhythmia_symbols = {
        'V',  # PVC (most common arrhythmia, 6.3%)
        'A',  # Atrial premature (2.3%)
        '/',  # Paced beat (6.2%)
        'F',  # Fusion ventricular + normal
        'f',  # Fusion paced + normal
        'E',  # Ventricular escape
        'J',  # Nodal premature
        'S',  # Supraventricular premature
        'a',  # Aberrated atrial
        'e',  # Atrial escape
        'j',  # Nodal escape
    }

    # Count arrhythmic beats
    arrhythmia_count = sum(1 for beat in beats if beat in arrhythmia_symbols)
    total_beats = len(beats)

    # Conservative labeling: ≥1 arrhythmic beat → Arrhythmia
    label = 1 if arrhythmia_count > 0 else 0

    # Confidence: fraction of arrhythmic beats
    confidence = arrhythmia_count / total_beats if total_beats > 0 else 0.0

    return label, confidence


def segment_signal(
    signal: np.ndarray,
    annotations: wfdb.Annotation,
    fs: int = 250,
    window_size: int = 2500,
    original_fs: int = 360
) -> List[Dict]:
    """
    Segment signal into fixed-length windows with annotations

    Args:
        signal: Input signal [n_samples] at target sampling rate
        annotations: wfdb.Annotation object (sample indices at original_fs)
        fs: Target sampling frequency (250 Hz)
        window_size: Segment size in samples (2500 = 10s @ 250Hz)
        original_fs: Original annotation sampling rate (360 Hz)

    Returns:
        List of segment dictionaries:
        {
            'signal': np.ndarray [window_size],
            'label': int (0=Normal, 1=Arrhythmia),
            'start_sample': int,
            'end_sample': int,
            'beats': List[str] (beat symbols in this segment),
            'confidence': float (fraction of arrhythmic beats),
            'sqi': float (signal quality score)
        }
    """
    segments = []
    n_samples = len(signal)
    n_windows = n_samples // window_size

    # Adjust annotation sample indices to target sampling rate
    # Annotations are at original_fs (360 Hz), need to scale to fs (250 Hz)
    annotation_samples_scaled = (annotations.sample * fs / original_fs).astype(int)

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        # Extract segment
        segment_signal = signal[start_idx:end_idx]

        # Find beats in this segment
        beat_mask = (annotation_samples_scaled >= start_idx) & (annotation_samples_scaled < end_idx)
        # Convert symbol array to numpy array and use boolean indexing
        symbol_array = np.array(annotations.symbol)
        beats_in_segment = symbol_array[beat_mask]

        # Skip segments with no beats (shouldn't happen, but defensive)
        if len(beats_in_segment) == 0:
            logger.warning(f"Segment {i} has no beats, skipping")
            continue

        # Map beats to segment label
        label, confidence = map_beat_to_segment(beats_in_segment.tolist())

        # Calculate SQI
        sqi = calculate_sqi(segment_signal, fs=fs)

        segments.append({
            'signal': segment_signal,
            'label': label,
            'start_sample': start_idx,
            'end_sample': end_idx,
            'beats': beats_in_segment.tolist(),
            'confidence': confidence,
            'sqi': sqi
        })

    return segments


def normalize_signal(
    signal: np.ndarray,
    method: str = 'zscore',
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize signal to reduce inter-patient variability

    Args:
        signal: Input signal (single segment or full patient recording)
        method: 'zscore', 'minmax', 'robust', or 'none'
        epsilon: Small constant to prevent division by zero

    Returns:
        Normalized signal

    Methods:
        zscore:  (x - mean) / std
        minmax:  (x - min) / (max - min)  → [0, 1]
        robust:  (x - median) / IQR  (outlier-resistant)
        none:    No normalization (raw signal)
    """
    if method == 'zscore':
        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / (std + epsilon)

    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized = (signal - min_val) / (max_val - min_val + epsilon)

    elif method == 'robust':
        median = np.median(signal)
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        normalized = (signal - median) / (iqr + epsilon)

    elif method == 'none':
        normalized = signal

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def time_warp(signal: np.ndarray, warp_factor: float) -> np.ndarray:
    """
    Apply time warping (temporal stretch/compress)

    Args:
        signal: Input signal
        warp_factor: 0.9-1.1 (±10% stretch)
                    <1.0 = compress (faster heart rate)
                    >1.0 = stretch (slower heart rate)

    Returns:
        Time-warped signal (resampled to original length)
    """
    n = len(signal)
    x_old = np.arange(n)
    x_new = np.linspace(0, n-1, int(n * warp_factor))

    interpolator = interp1d(x_old, signal, kind='cubic', fill_value='extrapolate')
    warped = interpolator(x_new)

    # Resample back to original length
    warped_resampled = sp_signal.resample(warped, n)
    return warped_resampled


def amplitude_scale(signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale signal amplitude (simulates gain variation)

    Args:
        signal: Input signal
        scale_factor: 0.8-1.2 (±20% amplitude)

    Returns:
        Scaled signal
    """
    return signal * scale_factor


def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise at specified SNR

    Args:
        signal: Input signal
        snr_db: Target SNR in dB (20-30 for mild noise)

    Returns:
        Noisy signal
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def add_baseline_wander(
    signal: np.ndarray,
    fs: int = 250,
    freq: float = 0.3,
    amplitude: float = 0.2
) -> np.ndarray:
    """
    Add synthetic baseline wander (breathing artifact)

    Args:
        signal: Input signal
        fs: Sampling frequency
        freq: Wander frequency 0.2-0.5 Hz (respiratory rate)
        amplitude: Wander amplitude relative to signal std

    Returns:
        Signal with baseline wander
    """
    t = np.arange(len(signal)) / fs
    wander = amplitude * np.std(signal) * np.sin(2 * np.pi * freq * t)
    return signal + wander


def augment_segment(
    signal: np.ndarray,
    fs: int = 250,
    augmentation_factor: int = 5
) -> List[np.ndarray]:
    """
    Generate augmented variants of a segment

    Args:
        signal: Original segment [window_size]
        fs: Sampling frequency
        augmentation_factor: Number of variants to generate

    Returns:
        List of augmented segments (includes original)
    """
    augmented = [signal]  # Always include original

    for i in range(augmentation_factor - 1):
        aug = signal.copy()

        # Randomly apply transformations (50% chance each)
        if np.random.rand() < 0.5:
            aug = time_warp(aug, warp_factor=np.random.uniform(0.9, 1.1))

        if np.random.rand() < 0.5:
            aug = amplitude_scale(aug, scale_factor=np.random.uniform(0.8, 1.2))

        if np.random.rand() < 0.5:
            aug = add_gaussian_noise(aug, snr_db=np.random.uniform(20, 30))

        if np.random.rand() < 0.3:  # Less frequent
            aug = add_baseline_wander(aug, fs=fs, freq=np.random.uniform(0.2, 0.5))

        augmented.append(aug)

    return augmented
