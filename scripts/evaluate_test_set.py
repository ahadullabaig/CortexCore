#!/usr/bin/env python3
"""
Test Set Evaluation Script
============================
Evaluates the trained model on the held-out test set (1000 samples)
to verify generalization performance after fixing the 100% accuracy issue.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleClassifier(nn.Module):
    """Simple classifier model (matching training script)"""
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def evaluate_test_set(
    model_path='models/best_model.pt',
    data_path='data/synthetic/test_data.pt',
    device=None,
    batch_size=16
):
    """Evaluate model on test set"""

    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print()

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    if device is None:
        device = get_device()
    device = torch.device(device)
    print(f"📱 Device: {device}")
    print()

    # Load test data
    print("📊 Loading test data...")
    test_data = torch.load(data_path)
    test_signals = test_data['signals']
    test_labels = test_data['labels']

    print(f"   ✅ Test samples: {len(test_signals)}")
    print(f"   ✅ Signal shape: {test_signals.shape}")
    print(f"   ✅ Class distribution:")
    n_normal = (test_labels == 0).sum().item()
    n_arrhythmia = (test_labels == 1).sum().item()
    print(f"      - Normal: {n_normal} ({n_normal/len(test_labels)*100:.1f}%)")
    print(f"      - Arrhythmia: {n_arrhythmia} ({n_arrhythmia/len(test_labels)*100:.1f}%)")
    print()

    # Create dataset and loader
    test_dataset = torch.utils.data.TensorDataset(test_signals, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Load model
    print("🧠 Loading model...")
    model = SimpleClassifier(
        input_size=test_signals.shape[1],
        num_classes=2
    )

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   ✅ Best validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("   ✅ Model loaded (state_dict only)")

    model.to(device)
    model.eval()
    print(f"   ✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Evaluate
    print("🧪 Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_labels = []
    all_losses = []

    correct = 0
    total = 0

    # Per-class metrics
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Predictions
            _, predicted = outputs.max(1)

            # Accumulate
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_losses.append(loss.item())

            # Overall accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for label in range(2):
                mask = labels == label
                if mask.sum() > 0:
                    class_correct[label] += predicted[mask].eq(labels[mask]).sum().item()
                    class_total[label] += mask.sum().item()

    # Compute metrics
    test_loss = np.mean(all_losses)
    test_accuracy = 100.0 * correct / total

    # Per-class accuracy
    normal_accuracy = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    arrhythmia_accuracy = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    # Confusion matrix
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    true_negatives = ((all_labels == 0) & (all_predictions == 0)).sum()
    false_positives = ((all_labels == 0) & (all_predictions == 1)).sum()
    false_negatives = ((all_labels == 1) & (all_predictions == 0)).sum()
    true_positives = ((all_labels == 1) & (all_predictions == 1)).sum()

    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print()
    print("=" * 60)
    print("✅ TEST SET RESULTS")
    print("=" * 60)
    print()

    print("📊 Overall Performance:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    print()

    print("📊 Per-Class Performance:")
    print(f"   Normal Accuracy: {normal_accuracy:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"   Arrhythmia Accuracy: {arrhythmia_accuracy:.2f}% ({class_correct[1]}/{class_total[1]})")
    print()

    print("📊 Classification Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1_score:.4f}")
    print()

    print("📊 Confusion Matrix:")
    print("              Predicted")
    print("              Normal  Arrhythmia")
    print(f"   Normal     {true_negatives:6d}  {false_positives:6d}")
    print(f"   Arrhythmia {false_negatives:6d}  {true_positives:6d}")
    print()

    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'normal_accuracy': float(normal_accuracy),
        'arrhythmia_accuracy': float(arrhythmia_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'confusion_matrix': {
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_positives': int(true_positives)
        },
        'total_samples': int(total),
        'correct_predictions': int(correct)
    }

    # Save to file
    results_dir = Path('results/metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'test_set_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {results_file}")
    print()

    # Interpretation
    print("=" * 60)
    print("📈 INTERPRETATION")
    print("=" * 60)
    print()

    if test_accuracy >= 85:
        print("✅ PASS: Test accuracy meets Phase 1 target (≥85%)")
    else:
        print("❌ FAIL: Test accuracy below Phase 1 target (≥85%)")

    if abs(test_accuracy - checkpoint.get('val_acc', 0)) < 5:
        print("✅ PASS: Test accuracy close to validation accuracy (good generalization)")
    else:
        print("⚠️  WARNING: Test accuracy differs significantly from validation accuracy")

    if test_accuracy < 100:
        print("✅ PASS: Test accuracy < 100% confirms realistic dataset")
    else:
        print("❌ FAIL: Test accuracy = 100% suggests dataset still too easy")

    print()
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        results = evaluate_test_set()
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
