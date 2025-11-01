"""
CortexCore System
==================================

Brain-inspired computing for medical diagnosis using Spiking Neural Networks.

Modules:
    - data: Data generation and preprocessing
    - model: SNN model architectures
    - train: Training pipelines
    - inference: Prediction and inference
    - utils: Helper utilities

Author: Your Team
License: MIT
"""

__version__ = "0.1.0-mvp"
__author__ = "CortexCore Team"

# Quick imports for convenience
try:
    from .data import generate_synthetic_ecg, load_dataset
    from .model import SimpleSNN
    from .inference import predict
    from .utils import set_seed, get_device
except ImportError:
    # Modules not yet implemented
    pass
