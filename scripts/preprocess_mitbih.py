#!/usr/bin/env python3
"""
MIT-BIH Arrhythmia Database Preprocessing Pipeline

Transforms raw MIT-BIH data into PyTorch-compatible datasets for SNN training.

Usage:
    python scripts/preprocess_mitbih.py
    python scripts/preprocess_mitbih.py --sqi_threshold 0.8
    python scripts/preprocess_mitbih.py --enable_augmentation

Output:
    data/mitbih_processed/
    ├── train_ecg.pt
    ├── val_ecg.pt
    ├── test_ecg.pt
    └── dataset_stats.json
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import json
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import (
    load_raw_record,
    resample_signal,
    filter_signal,
    normalize_signal,
    segment_signal,
    augment_segment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline"""
    # Input/Output
    input_dir: Path = Path('data/mitbih')
    output_dir: Path = Path('data/mitbih_processed')

    # Resampling
    original_fs: int = 360
    target_fs: int = 250

    # Filtering
    highpass_cutoff: float = 0.5
    lowpass_cutoff: float = 40.0
    notch_freq: float = 60.0
    filter_order: int = 4
    notch_q: float = 30.0

    # Segmentation
    window_size: int = 2500  # 10s @ 250Hz
    overlap: int = 0  # No overlap

    # Quality Control
    sqi_threshold: float = 0.7

    # Normalization
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust', 'none'

    # Patient Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    random_seed: int = 42

    # Augmentation (optional)
    enable_augmentation: bool = False
    augmentation_factor: int = 5

    # Logging
    verbose: bool = True


def find_patient_records(data_dir: Path) -> List[str]:
    """
    Find all patient record IDs in data directory

    Args:
        data_dir: Directory containing MIT-BIH files

    Returns:
        List of patient IDs (e.g., ['100', '101', ...])
    """
    hea_files = sorted(data_dir.glob('*.hea'))
    patient_ids = [f.stem for f in hea_files]
    logger.info(f"Found {len(patient_ids)} patient records")
    return patient_ids


def split_patients(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patients into train/val/test sets

    CRITICAL: Patient-based splitting prevents data leakage

    Args:
        patient_ids: List of patient IDs
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        random_seed: For reproducibility

    Returns:
        train_patients, val_patients, test_patients
    """
    np.random.seed(random_seed)
    patient_ids = np.array(patient_ids)
    np.random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = patient_ids[:n_train].tolist()
    val_patients = patient_ids[n_train:n_train + n_val].tolist()
    test_patients = patient_ids[n_train + n_val:].tolist()

    logger.info(f"Patient split: {len(train_patients)} train, "
                f"{len(val_patients)} val, {len(test_patients)} test")

    # Validation: Ensure no overlap
    assert len(set(train_patients) & set(val_patients)) == 0, "Train/Val overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, "Train/Test overlap!"
    assert len(set(val_patients) & set(test_patients)) == 0, "Val/Test overlap!"

    logger.info("✓ Patient split validation passed (no overlap)")

    return train_patients, val_patients, test_patients


def preprocess_single_patient(patient_id: str, config: PreprocessConfig) -> List[Dict]:
    """
    Full preprocessing pipeline for a single patient

    Steps:
        1. Load raw signal + annotations
        2. Extract Lead II (index 0)
        3. Resample 360 Hz → 250 Hz
        4. Filter (highpass, lowpass, notch)
        5. Normalize (per-patient Z-score)
        6. Segment into 10-second windows
        7. Map beat annotations to segment labels
        8. Calculate SQI for each segment
        9. (Optional) Augment segments

    Args:
        patient_id: Patient record ID
        config: Preprocessing configuration

    Returns:
        List of segment dictionaries

    Raises:
        Exception: If preprocessing fails for this patient
    """
    # Step 1: Load
    signal, annotations, record = load_raw_record(patient_id, config.input_dir)

    # Step 2: Use Lead II only (index 0)
    signal_lead2 = signal[:, 0]

    # Step 3: Resample
    signal_resampled = resample_signal(
        signal_lead2,
        original_fs=config.original_fs,
        target_fs=config.target_fs
    )

    # Step 4: Filter
    signal_filtered = filter_signal(
        signal_resampled,
        fs=config.target_fs,
        highpass_cutoff=config.highpass_cutoff,
        lowpass_cutoff=config.lowpass_cutoff,
        notch_freq=config.notch_freq,
        filter_order=config.filter_order,
        notch_q=config.notch_q
    )

    # Step 5: Normalize (per-patient)
    signal_normalized = normalize_signal(
        signal_filtered,
        method=config.normalization_method
    )

    # Step 6-8: Segment, annotate, calculate SQI
    segments = segment_signal(
        signal=signal_normalized,
        annotations=annotations,
        fs=config.target_fs,
        window_size=config.window_size,
        original_fs=config.original_fs
    )

    # Step 9: Augmentation (optional)
    if config.enable_augmentation:
        augmented_segments = []
        for segment in segments:
            # Generate augmented variants
            augmented_signals = augment_segment(
                segment['signal'],
                fs=config.target_fs,
                augmentation_factor=config.augmentation_factor
            )

            # Create segment dict for each variant
            for aug_signal in augmented_signals:
                aug_segment = segment.copy()
                aug_segment['signal'] = aug_signal
                augmented_segments.append(aug_segment)

        segments = augmented_segments

    return segments


def create_pytorch_datasets(
    segments_by_patient: Dict[str, List[Dict]],
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    output_dir: Path,
    sqi_threshold: float = 0.7
) -> Dict[str, Dict]:
    """
    Create train/val/test PyTorch datasets compatible with ECGDataset

    Args:
        segments_by_patient: Dict mapping patient_id → List[segment_dict]
        train/val/test_patients: Patient ID lists
        output_dir: Where to save .pt files
        sqi_threshold: Reject segments with SQI < threshold

    Returns:
        Dataset statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def collect_segments(patient_list: List[str]) -> Dict:
        """Collect segments from list of patients"""
        all_signals = []
        all_labels = []
        all_patient_ids = []
        all_confidence = []
        all_sqi = []

        for patient_id in patient_list:
            if patient_id not in segments_by_patient:
                logger.warning(f"Patient {patient_id} not in segments dict, skipping")
                continue

            segments = segments_by_patient[patient_id]

            for segment in segments:
                # Quality control: reject low SQI
                if segment['sqi'] < sqi_threshold:
                    continue

                all_signals.append(segment['signal'])
                all_labels.append(segment['label'])
                all_patient_ids.append(patient_id)
                all_confidence.append(segment['confidence'])
                all_sqi.append(segment['sqi'])

        return {
            'signals': np.array(all_signals, dtype=np.float32),
            'labels': np.array(all_labels, dtype=np.int64),
            'patient_ids': all_patient_ids,
            'segment_confidence': np.array(all_confidence, dtype=np.float32),
            'sqi_scores': np.array(all_sqi, dtype=np.float32)
        }

    # Collect datasets
    logger.info("Collecting train segments...")
    train_data = collect_segments(train_patients)

    logger.info("Collecting validation segments...")
    val_data = collect_segments(val_patients)

    logger.info("Collecting test segments...")
    test_data = collect_segments(test_patients)

    # Save to disk
    logger.info("Saving datasets...")
    torch.save(train_data, output_dir / 'train_ecg.pt')
    torch.save(val_data, output_dir / 'val_ecg.pt')
    torch.save(test_data, output_dir / 'test_ecg.pt')

    # Calculate statistics
    stats = {}
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        n_patients = len(set(data['patient_ids']))
        n_segments = len(data['labels'])
        normal_count = int((data['labels'] == 0).sum())
        arrhythmia_count = int((data['labels'] == 1).sum())

        stats[split_name] = {
            'n_segments': n_segments,
            'n_patients': n_patients,
            'class_distribution': {
                'normal': normal_count,
                'arrhythmia': arrhythmia_count,
                'normal_pct': float(normal_count / n_segments * 100) if n_segments > 0 else 0.0,
                'arrhythmia_pct': float(arrhythmia_count / n_segments * 100) if n_segments > 0 else 0.0
            },
            'mean_sqi': float(data['sqi_scores'].mean()) if len(data['sqi_scores']) > 0 else 0.0,
            'segments_per_patient': float(n_segments / n_patients) if n_patients > 0 else 0.0
        }

    # Save stats as JSON
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("DATASET CREATION SUMMARY")
    print("="*80)
    for split in ['train', 'val', 'test']:
        s = stats[split]
        print(f"\n{split.upper()}:")
        print(f"  Patients: {s['n_patients']}")
        print(f"  Segments: {s['n_segments']}")
        print(f"  Normal: {s['class_distribution']['normal']} "
              f"({s['class_distribution']['normal_pct']:.1f}%)")
        print(f"  Arrhythmia: {s['class_distribution']['arrhythmia']} "
              f"({s['class_distribution']['arrhythmia_pct']:.1f}%)")
        print(f"  Mean SQI: {s['mean_sqi']:.3f}")
        print(f"  Segments/Patient: {s['segments_per_patient']:.1f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIT-BIH Arrhythmia Database')
    parser.add_argument('--input_dir', type=str, default='data/mitbih',
                        help='Input directory with MIT-BIH files')
    parser.add_argument('--output_dir', type=str, default='data/mitbih_processed',
                        help='Output directory for processed datasets')
    parser.add_argument('--sqi_threshold', type=float, default=0.7,
                        help='SQI threshold for quality control')
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=5,
                        help='Augmentation factor (if enabled)')
    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        sqi_threshold=args.sqi_threshold,
        enable_augmentation=args.enable_augmentation,
        augmentation_factor=args.augmentation_factor
    )

    print("="*80)
    print("MIT-BIH ARRHYTHMIA DATABASE PREPROCESSING")
    print("="*80)
    print(f"Input:  {config.input_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Target sampling rate: {config.target_fs} Hz")
    print(f"Segment size: {config.window_size} samples (10 seconds)")
    print(f"SQI threshold: {config.sqi_threshold}")
    print(f"Augmentation: {'Enabled (factor=' + str(config.augmentation_factor) + ')' if config.enable_augmentation else 'Disabled'}")
    print("="*80)

    # Step 1: Find all patient records
    try:
        patient_ids = find_patient_records(config.input_dir)
        if len(patient_ids) == 0:
            logger.error("No patient records found! Check input directory.")
            return 1
    except Exception as e:
        logger.error(f"Error finding patient records: {e}")
        return 1

    # Step 2: Split patients (before preprocessing to ensure clean splits)
    train_patients, val_patients, test_patients = split_patients(
        patient_ids,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        random_seed=config.random_seed
    )

    # Step 3: Preprocess all patients
    print("\n" + "="*80)
    print("PREPROCESSING PATIENTS")
    print("="*80)
    segments_by_patient = {}
    failed_patients = []

    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        try:
            segments = preprocess_single_patient(patient_id, config)
            segments_by_patient[patient_id] = segments
        except Exception as e:
            logger.error(f"ERROR processing patient {patient_id}: {e}")
            failed_patients.append(patient_id)
            continue

    print(f"\n✅ Preprocessing complete: {len(segments_by_patient)}/{len(patient_ids)} patients successful")
    if failed_patients:
        print(f"⚠️  Failed patients: {failed_patients}")

    # Step 4: Create PyTorch datasets
    print("\n" + "="*80)
    print("CREATING PYTORCH DATASETS")
    print("="*80)
    try:
        stats = create_pytorch_datasets(
            segments_by_patient=segments_by_patient,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            output_dir=config.output_dir,
            sqi_threshold=config.sqi_threshold
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return 1

    # Final summary
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nDatasets saved to: {config.output_dir}")
    print(f"Statistics saved to: {config.output_dir / 'dataset_stats.json'}")
    print("\nNext steps:")
    print("  1. Review dataset_stats.json for class distribution")
    print("  2. Start transfer learning training:")
    print(f"     python scripts/train_mitbih_transfer.py --data_dir {config.output_dir}")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPreprocessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
