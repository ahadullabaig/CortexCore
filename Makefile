# ============================================
# CortexCore - Makefile
# ============================================
# Common commands for development and deployment
# Usage: make <command>

.PHONY: help install clean test train evaluate demo docker format lint notebook

# Default target
.DEFAULT_GOAL := help

# ============================================
# Help
# ============================================

help:  ## Show this help message
	@echo "🧠 CortexCore - Available Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Setup & Installation
# ============================================

install:  ## Install all dependencies
	@echo "📦 Installing dependencies..."
	bash scripts/01_setup_environment.sh

install-dev:  ## Install with development dependencies
	@echo "📦 Installing with dev dependencies..."
	pip install -e ".[dev]"

install-deploy:  ## Install with deployment dependencies
	@echo "📦 Installing with deployment dependencies..."
	pip install -e ".[deployment]"

# ============================================
# Data
# ============================================

generate-data:  ## Generate synthetic dataset
	@echo "📊 Generating synthetic data..."
	bash scripts/02_generate_mvp_data.sh

preprocess:  ## Run preprocessing pipeline
	@echo "🔧 Preprocessing data..."
	python src/data.py

# ============================================
# Training
# ============================================

train:  ## Train model with default config
	@echo "🏋️  Training model..."
	bash scripts/03_train_mvp_model.sh

train-debug:  ## Train with debug mode (verbose output)
	@echo "🐛 Training in debug mode..."
	python src/train.py --debug --log-level DEBUG

train-fast:  ## Quick training run (5 epochs)
	@echo "⚡ Fast training..."
	python src/train.py --epochs 5

# ============================================
# Evaluation
# ============================================

evaluate:  ## Evaluate trained model
	@echo "📊 Evaluating model..."
	python src/inference.py --evaluate --model models/best_model.pt

metrics:  ## Calculate clinical metrics
	@echo "📈 Calculating metrics..."
	python -c "from src.utils import calculate_clinical_metrics; print('Metrics calculation')"

visualize:  ## Generate visualizations
	@echo "📊 Generating visualizations..."
	python scripts/visualize.py

# ============================================
# Demo
# ============================================

demo:  ## Launch demo server
	@echo "🎬 Launching demo..."
	bash scripts/04_run_demo.sh

demo-production:  ## Launch production demo
	@echo "🚀 Launching production demo..."
	gunicorn -w 4 -b 0.0.0.0:5000 demo.app:app

# ============================================
# Testing
# ============================================

test:  ## Run all tests
	@echo "🧪 Running tests..."
	bash scripts/05_test_integration.sh

test-unit:  ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-coverage:  ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html

# ============================================
# Code Quality
# ============================================

format:  ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black src/ tests/ demo/
	isort src/ tests/ demo/

lint:  ## Lint code with flake8
	@echo "🔍 Linting code..."
	flake8 src/ tests/ demo/ --max-line-length=120

type-check:  ## Type checking with mypy
	@echo "🔍 Type checking..."
	mypy src/ --ignore-missing-imports

check:  ## Run all checks (format, lint, type-check)
	@echo "✅ Running all checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check

# ============================================
# Docker
# ============================================

docker-build:  ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker-compose build

docker-run:  ## Run Docker containers
	@echo "🐳 Running Docker containers..."
	docker-compose up

docker-stop:  ## Stop Docker containers
	@echo "🛑 Stopping Docker containers..."
	docker-compose down

docker-clean:  ## Clean Docker images and containers
	@echo "🧹 Cleaning Docker..."
	docker-compose down -v
	docker system prune -f

# ============================================
# Development
# ============================================

notebook:  ## Launch Jupyter notebook
	@echo "📓 Launching Jupyter..."
	jupyter notebook notebooks/

tensorboard:  ## Launch TensorBoard
	@echo "📊 Launching TensorBoard..."
	tensorboard --logdir=experiments/logs

# ============================================
# Benchmarking
# ============================================

benchmark:  ## Run performance benchmarks
	@echo "⚡ Running benchmarks..."
	python scripts/benchmark.py --all

benchmark-cpu:  ## Benchmark CPU performance
	@echo "⚡ Benchmarking CPU..."
	python scripts/benchmark.py --device cpu

benchmark-gpu:  ## Benchmark GPU performance
	@echo "⚡ Benchmarking GPU..."
	python scripts/benchmark.py --device cuda

# ============================================
# Deployment
# ============================================

export-onnx:  ## Export model to ONNX format
	@echo "📦 Exporting to ONNX..."
	python -m src.deployment.export_onnx --model models/best_model.pt

optimize-model:  ## Optimize model for deployment
	@echo "⚡ Optimizing model..."
	python -m src.deployment.optimize

deploy-edge:  ## Deploy to edge device
	@echo "🚀 Deploying to edge..."
	python scripts/deploy.py --target edge

# ============================================
# Cleaning
# ============================================

clean:  ## Clean temporary files
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete

clean-data:  ## Clean generated data (WARNING: deletes all data)
	@echo "🧹 Cleaning data..."
	rm -rf data/synthetic/*.pt
	rm -rf data/cache/*

clean-models:  ## Clean trained models (WARNING: deletes all models)
	@echo "🧹 Cleaning models..."
	rm -rf models/*.pt
	rm -rf models/checkpoints/*

clean-all:  ## Clean everything (WARNING: destructive)
	@echo "🧹 Cleaning everything..."
	$(MAKE) clean
	$(MAKE) clean-data
	$(MAKE) clean-models
	rm -rf experiments/
	rm -rf results/
	rm -rf logs/

# ============================================
# Documentation
# ============================================

docs:  ## Build documentation
	@echo "📚 Building documentation..."
	cd docs && make html

docs-serve:  ## Serve documentation locally
	@echo "📚 Serving documentation..."
	cd docs/_build/html && python -m http.server 8000

# ============================================
# Git Helpers
# ============================================

status:  ## Show git status
	@git status

commit:  ## Quick commit (use: make commit msg="your message")
	@git add .
	@git commit -m "$(msg)"

push:  ## Push to remote
	@git push origin main

pull:  ## Pull from remote
	@git pull origin main

# ============================================
# Monitoring
# ============================================

monitor:  ## Monitor system resources
	@echo "📊 Monitoring system..."
	watch -n 1 'nvidia-smi || echo "No GPU available"'

logs:  ## Tail application logs
	@echo "📋 Tailing logs..."
	tail -f logs/app.log

# ============================================
# Quick Shortcuts
# ============================================

quick-start: install generate-data train demo  ## Full quick start pipeline

day1: install generate-data  ## Day 1 setup

day3: test  ## Day 3 integration check

demo-day: clean-all install generate-data train demo  ## Prepare for demo day

# ============================================
# Info
# ============================================

info:  ## Show project information
	@echo "🧠 CortexCore"
	@echo ""
	@echo "📂 Project Structure:"
	@ls -la
	@echo ""
	@echo "🐍 Python Version:"
	@python --version
	@echo ""
	@echo "📦 Installed Packages:"
	@pip list | grep -E "(torch|snntorch|flask|neurokit2)" || echo "Packages not installed yet"
	@echo ""
	@echo "💾 Disk Usage:"
	@du -sh data/ models/ 2>/dev/null || echo "No data/models yet"
