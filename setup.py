"""
Neuromorphic SNN Healthcare - Package Setup
==========================================

Installation:
    pip install -e .          # Development mode
    pip install .             # Standard installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="neuromorphic-snn-healthcare",
    version="0.1.0",
    author="Neuromorphic Healthcare Team",
    author_email="team@neuromorphic-health.ai",
    description="Brain-inspired computing for medical diagnosis using Spiking Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-team/neuromorphic-snn-healthcare",
    packages=find_packages(exclude=["tests*", "notebooks*", "demo*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "deployment": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "tensorrt>=8.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-train=src.train:main",
            "snn-predict=src.inference:main",
            "snn-demo=demo.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
