"""
Utility Functions Module
========================

Owner: CS1 / Team Lead

Responsibilities:
- Common helper functions
- Logging utilities
- Device management
- Reproducibility

Phase: Days 1-30
"""

import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Optional, Union, Dict
import json
import os

# ============================================
# Reproducibility
# ============================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================
# Device Management
# ============================================

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device (CUDA if available, else CPU)

    Args:
        device: Specific device string ('cuda', 'cpu', 'mps')

    Returns:
        PyTorch device
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def print_device_info():
    """Print information about available devices"""
    print("🖥️  Device Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    print(f"   MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"   Default device: {get_device()}")


# ============================================
# Logging
# ============================================

def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_file: Path to log file (optional)
        log_level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger('cortexcore')
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


# ============================================
# Configuration Management
# ============================================

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON/YAML file"""
    path = Path(config_path)

    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict, config_path: str):
    """Save configuration to file"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================
# File System Utilities
# ============================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def list_files(directory: str, pattern: str = '*') -> list:
    """List files matching pattern in directory"""
    return list(Path(directory).glob(pattern))


# ============================================
# Metrics & Statistics
# ============================================

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy"""
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def calculate_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Calculate confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1
    return cm


def calculate_clinical_metrics(confusion_matrix: np.ndarray, class_idx: int = 1) -> Dict[str, float]:
    """
    Calculate clinical validation metrics (sensitivity, specificity, etc.)

    Args:
        confusion_matrix: Confusion matrix
        class_idx: Index of positive class

    Returns:
        Clinical metrics dictionary
    """
    tp = confusion_matrix[class_idx, class_idx]
    fn = confusion_matrix[class_idx, :].sum() - tp
    fp = confusion_matrix[:, class_idx].sum() - tp
    tn = confusion_matrix.sum() - tp - fn - fp

    metrics = {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    }

    return metrics


# ============================================
# Formatting & Display
# ============================================

def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty print metrics dictionary"""
    print(f"{prefix}Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


# ============================================
# Environment Variables
# ============================================

def load_env(env_file: str = '.env'):
    """Load environment variables from file"""
    if not Path(env_file).exists():
        return

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with default"""
    return os.getenv(key, default)


# ============================================
# Example Usage & Testing
# ============================================

if __name__ == "__main__":
    print("🧪 Testing utils module...")

    # Test device info
    print("\n1. Device Information:")
    print_device_info()

    # Test seed setting
    print("\n2. Setting random seed...")
    set_seed(42)
    print("   ✅ Seed set to 42")

    # Test metrics
    print("\n3. Testing metrics...")
    predictions = torch.tensor([0, 1, 1, 0, 1])
    targets = torch.tensor([0, 1, 0, 0, 1])
    acc = calculate_accuracy(predictions, targets)
    print(f"   Accuracy: {acc:.2f}%")

    # Test confusion matrix
    cm = calculate_confusion_matrix(predictions.numpy(), targets.numpy(), num_classes=2)
    print(f"   Confusion Matrix:\n{cm}")

    # Test clinical metrics
    clinical = calculate_clinical_metrics(cm)
    print_metrics(clinical, "   Clinical ")

    print("\n✅ Utils module working!")
