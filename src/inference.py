"""
Inference & Prediction Module
==============================

Owner: CS2 + CS4

Responsibilities:
- Model inference
- Prediction functions
- Performance profiling
- Batch processing

Phase: Days 4-30
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union, List, Optional
import time
from pathlib import Path

# ============================================
# TODO: Day 4-5 - Basic Inference
# ============================================

def load_model(model_path: str, model_class: nn.Module, device: str = 'cuda') -> nn.Module:
    """
    Load trained model from checkpoint

    Args:
        model_path: Path to model checkpoint
        model_class: Model class instance
        device: Device to load model on

    Returns:
        Loaded model

    TODO:
        - Add model versioning
        - Support multiple model formats (ONNX, TorchScript)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model_class.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_class.load_state_dict(checkpoint)

    model_class.eval()
    model_class.to(device)

    return model_class


def predict(
    model: nn.Module,
    input_data: Union[torch.Tensor, np.ndarray],
    device: str = 'cuda',
    return_confidence: bool = True,
    num_steps: int = 100,
    gain: float = 10.0,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[int, float, np.ndarray, str]]:
    """
    Make prediction on input data

    Args:
        model: Trained model
        input_data: Input signal [signal_length] or [batch, signal_length]
        device: Device for inference
        return_confidence: Whether to return confidence scores
        num_steps: Number of time steps for SNN (replicate signal)
        class_names: Optional class names for output

    Returns:
        Prediction dictionary with prediction, confidence, class_name, etc.
    """
    if class_names is None:
        class_names = ['Normal', 'Arrhythmia']

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # Convert to numpy if tensor
    if isinstance(input_data, torch.Tensor):
        signal_np = input_data.cpu().numpy()
    else:
        signal_np = input_data

    # Handle single signal vs batch and apply 2D spike encoding
    if len(signal_np.shape) == 1:
        # Single signal: [signal_length] -> [num_steps, 1, signal_length]
        # Normalize signal
        signal_norm = (signal_np - signal_np.min()) / (signal_np.max() - signal_np.min() + 1e-8)
        # Replicate across time steps and apply Poisson encoding
        spikes = np.random.rand(num_steps, len(signal_np)) < (signal_norm * gain / 100.0)
        input_data = torch.FloatTensor(spikes).unsqueeze(1)  # [num_steps, 1, signal_length]
    elif len(signal_np.shape) == 2:
        # Batch of signals: [batch, signal_length] -> [num_steps, batch, signal_length]
        batch_spikes = []
        for i in range(signal_np.shape[0]):
            signal_norm = (signal_np[i] - signal_np[i].min()) / (signal_np[i].max() - signal_np[i].min() + 1e-8)
            spikes = np.random.rand(num_steps, len(signal_np[i])) < (signal_norm * gain / 100.0)
            batch_spikes.append(spikes)
        input_data = torch.FloatTensor(np.stack(batch_spikes, axis=1))  # [num_steps, batch, signal_length]
    elif len(signal_np.shape) == 3:
        # Already encoded: [num_steps, batch, signal_length]
        input_data = torch.FloatTensor(signal_np)
    else:
        raise ValueError(f"Unexpected input shape: {signal_np.shape}")

    input_data = input_data.to(device)

    # Time inference
    start_time = time.time()

    with torch.no_grad():
        output = model(input_data)

        # Handle SNN-specific output (spikes + membrane + optional intermediate)
        if isinstance(output, tuple):
            if len(output) == 3:
                # HybridSTDP_SNN: (spikes, membrane, intermediate)
                spikes, membrane, intermediate = output
            else:
                # SimpleSNN: (spikes, membrane)
                spikes, membrane = output
            # Use spike counts for prediction
            output = spikes.sum(dim=0)  # Sum over time: [batch, classes]
            spike_count = spikes.sum().item()
        else:
            spike_count = 0

        # Get prediction
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1)
        confidence = probabilities.max(dim=1).values

    inference_time = (time.time() - start_time) * 1000  # ms

    pred_idx = prediction.cpu().item()
    result = {
        'prediction': pred_idx,
        'class_name': class_names[pred_idx],
        'inference_time_ms': inference_time,
        'spike_count': spike_count
    }

    if return_confidence:
        result['confidence'] = confidence.cpu().item()
        result['probabilities'] = probabilities.cpu().numpy()[0].tolist()

    return result


# ============================================
# TODO: Day 5-7 - Batch Inference
# ============================================

def batch_predict(
    model: nn.Module,
    inputs: Union[torch.Tensor, np.ndarray],
    batch_size: int = 32,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Batch prediction for multiple inputs

    TODO:
        - Optimize for throughput
        - Add progress tracking
        - Implement streaming inference
    """
    results = []

    n_samples = len(inputs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = inputs[start_idx:end_idx]

        batch_results = predict(model, batch, device)
        results.append(batch_results)

    return results


# ============================================
# TODO: Day 8+ - Advanced Features
# ============================================

def real_time_inference():
    """TODO: Implement streaming inference"""
    raise NotImplementedError("Week 2 task")


def ensemble_predict():
    """TODO: Implement ensemble predictions"""
    raise NotImplementedError("Week 3 task")


def explain_prediction():
    """TODO: Implement interpretability features"""
    raise NotImplementedError("Week 3 task")


# ============================================
# Performance Profiling
# ============================================

def profile_inference(
    model: nn.Module,
    input_shape: tuple,
    n_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Profile model inference performance

    Args:
        model: Model to profile
        input_shape: Input tensor shape
        n_iterations: Number of iterations for averaging
        device: Device for profiling

    Returns:
        Performance metrics

    TODO:
        - Add memory profiling
        - Profile on multiple devices
        - Measure energy consumption
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # Warmup
    dummy_input = torch.randn(*input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Profile
    times = []
    for _ in range(n_iterations):
        dummy_input = torch.randn(*input_shape).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            output = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.time() - start_time) * 1000  # ms
        times.append(elapsed)

    metrics = {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'median_time_ms': np.median(times),
        'throughput_samples_per_sec': 1000.0 / np.mean(times)
    }

    return metrics


# ============================================
# Clinical Validation Utilities
# ============================================

def clinical_prediction(
    model: nn.Module,
    signal: np.ndarray,
    class_names: List[str] = ['Normal', 'Arrhythmia'],
    threshold: float = 0.5
) -> Dict:
    """
    Make clinical prediction with safety checks

    TODO: Day 6-7
        - Add confidence thresholds
        - Implement flagging for uncertain cases
        - Add clinical interpretation
    """
    result = predict(model, signal)

    prediction = {
        'class': class_names[result['prediction']],
        'confidence': result['confidence'],
        'inference_time_ms': result['inference_time_ms'],
        'clinical_flag': 'REVIEW' if result['confidence'] < threshold else 'OK'
    }

    return prediction


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 Testing inference module...")
    print("✅ Inference module loaded successfully")
    print("\n📝 To use:")
    print("   1. Load model: model = load_model('models/best_model.pt', model_class)")
    print("   2. Predict: result = predict(model, input_data)")
    print("   3. Profile: metrics = profile_inference(model, input_shape)")
