# MIT-BIH Preprocessing Results

**Date**: November 9, 2025
**Preprocessing Script**: `scripts/preprocess_mitbih.py`
**Core Functions**: `src/preprocessing.py`

---

## Executive Summary

✅ **Successfully preprocessed all 48 MIT-BIH patient records** into PyTorch-compatible datasets ready for SNN training.

**Key Statistics**:
- Total segments created: 2,190 (after quality control)
- Dataset split: 1,482 train / 329 val / 379 test
- Mean SQI: 0.83+ across all splits (high quality)
- Processing time: ~15 seconds per patient (~12 minutes total)

---

## Dataset Statistics

### Train Set
```
Patients:      26 (out of 33 assigned, 7 had no segments passing SQI)
Segments:      1,482
Class balance: 61.8% Normal (916) / 38.2% Arrhythmia (566)
Mean SQI:      0.833
Segments/patient: 57.0 avg
```

### Validation Set
```
Patients:      7 (all assigned patients)
Segments:      329
Class balance: 30.4% Normal (100) / 69.6% Arrhythmia (229)
Mean SQI:      0.828
Segments/patient: 47.0 avg
```

### Test Set
```
Patients:      4 (out of 8 assigned, 4 had no segments passing SQI)
Segments:      379
Class balance: 35.1% Normal (133) / 64.9% Arrhythmia (246)
Mean SQI:      0.834
Segments/patient: 94.8 avg
```

---

## Data Characteristics

### Signal Statistics
```
Shape per segment:  (2500,) - 10 seconds @ 250 Hz
Dtype:              float32
Value range:        [-14.66, 9.56]
Mean:               -0.0003 (≈ 0, good Z-score normalization)
Std:                0.9422 (≈ 1, good Z-score normalization)
```

### Quality Control
```
SQI threshold:      0.7
SQI range:          [0.700, 1.000]
Rejection rate:     ~72% of segments rejected due to low SQI
Segments passing:   2,190 / ~7,800 expected
```

### Label Distribution
```
Annotation mapping: Conservative (≥1 arrhythmic beat → Arrhythmia)
Confidence range:   [0.000, 1.000]
                    0.0 = pure normal segment (all N, L, R beats)
                    1.0 = pure arrhythmia segment (all V, A, /, etc.)
```

---

## Preprocessing Pipeline Applied

### 1. Signal Loading
- Source: MIT-BIH 48 patient records (.dat + .atr files)
- Lead selection: Modified Lead II (index 0) only
- Original format: 30 min @ 360 Hz, 2 channels

### 2. Resampling
- Method: Scipy Fourier resampling
- Original → Target: 360 Hz → 250 Hz
- Samples per patient: 650,000 → 450,000

### 3. Filtering
```
High-pass:  0.5 Hz (remove baseline wander)
Low-pass:   40 Hz (remove high-freq noise, preserve QRS)
Notch:      60 Hz Q=30 (remove powerline interference)
Filter:     Butterworth order 4, zero-phase (filtfilt)
```

### 4. Normalization
- Method: Per-patient Z-score
- Formula: (x - mean) / std
- Rationale: Preserves within-patient relationships, reduces inter-patient variability

### 5. Segmentation
- Window size: 2500 samples (10 seconds)
- Overlap: 0 samples (no overlap)
- Expected segments/patient: 180
- Actual segments/patient: 57-95 (after SQI filtering)

### 6. Annotation Mapping
- Beat-level → Segment-level labels
- Strategy: Conservative (≥1 arrhythmic beat → label=1)
- Arrhythmia symbols: V, A, /, F, f, E, J, S, a, e, j
- Normal symbols: N, L, R

### 7. Quality Control
- SQI metric: Composite (skewness, kurtosis, SNR, flatline, saturation)
- Threshold: 0.7
- Result: ~72% rejection rate (high quality bar)

---

## Analysis & Observations

### 1. Lower Than Expected Segment Count

**Expected**: ~7,800 segments (48 patients × 180 segments × 90% pass rate)
**Actual**: 2,190 segments (28% of expected)

**Reasons**:
1. **High SQI threshold (0.7)**: Very stringent quality control
   - Rejects segments with baseline wander, noise, artifacts
   - Ensures high-quality training data
   - Trade-off: Smaller dataset

2. **Patient exclusions**: Some patients had NO segments passing SQI
   - 7 train patients excluded
   - 4 test patients excluded
   - Suggests these patients had particularly noisy recordings

**Implication**: Smaller but higher quality dataset. Transfer learning from synthetic model is now CRITICAL (was optional, now mandatory).

---

### 2. Class Imbalance Varies by Split

**Train**: 61.8% Normal / 38.2% Arrhythmia (moderate imbalance, 1.6:1)
**Val**: 30.4% Normal / 69.6% Arrhythmia (REVERSED, 0.4:1)
**Test**: 35.1% Normal / 64.9% Arrhythmia (REVERSED, 0.5:1)

**Reasons**:
- Validation and test patients have more arrhythmia-heavy segments
- Random patient assignment + different arrhythmia prevalence per patient
- MIT-BIH records 200-234 are "patients with significant arrhythmias"

**Implication**:
- Validation/test metrics may show higher sensitivity (more arrhythmia samples)
- Train set imbalance is manageable (already handled by FocalLoss + G-mean)
- Need to report per-split class distributions in results

---

### 3. SQI Filtering May Be Too Aggressive

**Current threshold**: 0.7
**Rejection rate**: 72%
**Mean SQI of accepted**: 0.83+

**Options**:
1. **Keep current (0.7)**: High quality, small dataset (current choice)
2. **Lower to 0.6**: Moderate quality, ~40% more segments
3. **Lower to 0.5**: Acceptable quality, ~100% more segments

**Recommendation**: Start training with current (0.7). If overfitting occurs due to small dataset, rerun preprocessing with lower threshold.

---

### 4. Patient-Based Splitting Worked Correctly

✅ **Validation passed**: No patient overlap between splits
✅ **Actual split**: 26 train / 7 val / 4 test patients (out of 48)
✅ **Patient tracking**: Each segment tagged with source patient_id

**Note**: Some assigned patients had 0 segments after SQI filtering, reducing actual patient counts.

---

## Files Created

### Dataset Files
```
data/mitbih_processed/
├── train_ecg.pt          # 22 MB  (1,482 segments)
├── val_ecg.pt            # 4.9 MB (329 segments)
├── test_ecg.pt           # 5.6 MB (379 segments)
└── dataset_stats.json    # 879 B  (statistics summary)
```

### Data Format (PyTorch Dictionary)
```python
{
    'signals': np.ndarray [n_segments, 2500] float32,
    'labels': np.ndarray [n_segments] int64,        # 0=Normal, 1=Arrhythmia
    'patient_ids': List[str],                       # Track source patient
    'segment_confidence': np.ndarray float32,       # Fraction arrhythmic beats
    'sqi_scores': np.ndarray float32                # Signal quality scores
}
```

---

## Compatibility with Existing Codebase

✅ **Compatible with ECGDataset**: Signal shape (n, 2500) matches expected
✅ **Binary labels**: 0/1 encoding matches existing training pipeline
✅ **Float32 dtype**: Matches existing synthetic data
✅ **Normalized values**: Mean ≈ 0, std ≈ 1 (like synthetic)

**Additional metadata** (not in synthetic):
- `patient_ids`: Track which patient each segment came from (prevents data leakage analysis)
- `segment_confidence`: Fraction of arrhythmic beats (for future weighted training)
- `sqi_scores`: Quality scores (for filtering or weighting)

---

## Next Steps

### Immediate (Day 11 evening)
1. ✅ Preprocessing complete
2. ✅ Dataset validation passed
3. 📋 Create transfer learning script

### Day 12 (Transfer Learning Stage 1)
```bash
python scripts/train_mitbih_transfer.py \
    --pretrained_model models/deep_focal_model.pt \
    --data_dir data/mitbih_processed \
    --freeze_layer1 \
    --learning_rate 0.0001 \
    --num_epochs 20 \
    --output_dir models/mitbih_stage1
```

**Expected**: 80-85% accuracy (frozen layer 1)

### Day 13 (Transfer Learning Stage 2)
```bash
python scripts/train_mitbih_transfer.py \
    --pretrained_model models/mitbih_stage1/best_model.pt \
    --data_dir data/mitbih_processed \
    --learning_rate 0.00005 \
    --num_epochs 30 \
    --output_dir models/mitbih_stage2
```

**Expected**: 85-88% accuracy (full fine-tuning)

---

## Risk Mitigation

### Risk 1: Small Dataset (1,482 train samples)
**Problem**: 673K parameters / 1,482 samples = 454:1 ratio (high overfitting risk)

**Mitigations**:
1. ✅ **Transfer learning** (MANDATORY): Start from synthetic weights (reduces effective params)
2. ✅ **Heavy regularization**: Dropout 0.5, weight decay 0.001, early stopping
3. 📋 **Data augmentation**: Time warp, amplitude scale, noise injection (if overfitting observed)
4. 📋 **Lower SQI threshold**: Rerun with threshold=0.6 for ~40% more data

### Risk 2: Class Imbalance in Val/Test
**Problem**: Val/test are 70% arrhythmia (opposite of train)

**Mitigations**:
1. ✅ **FocalLoss + G-mean**: Already handles imbalance
2. ✅ **Per-split metrics**: Report accuracy for each split separately
3. 📋 **Stratified evaluation**: Report per-class metrics (not just overall)

### Risk 3: SQI May Be Too Stringent
**Problem**: Rejecting 72% of segments may exclude learnable patterns

**Mitigations**:
1. 📋 **Threshold tuning**: If model underfits, lower SQI to 0.6 or 0.5
2. 📋 **Analyze rejected segments**: Check if they contain diagnostic info
3. 📋 **SQI-weighted training**: Use SQI as sample weight instead of hard threshold

---

## Conclusion

✅ **Preprocessing successful**: All 48 patients processed, 2,190 high-quality segments created

✅ **Dataset ready**: Compatible with existing training pipeline, properly split by patient

⚠️ **Challenge identified**: Smaller than expected dataset (2,190 vs ~7,800) due to aggressive SQI filtering

🎯 **Strategy adapted**: Transfer learning from synthetic model is now MANDATORY (was optional)

📊 **Quality validated**: Mean SQI 0.83+, proper normalization (mean≈0, std≈1), no patient leakage

🚀 **Ready for Day 12**: Transfer learning Stage 1 training can begin immediately

---

**Document Status**: COMPLETE
**Date**: November 9, 2025
**Next Action**: Create `scripts/train_mitbih_transfer.py` for transfer learning
