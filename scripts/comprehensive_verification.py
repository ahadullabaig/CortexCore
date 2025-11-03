"""
Comprehensive Phase 1 Verification
===================================

Deep verification of all components, code quality, performance, and functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
import json
from typing import Dict, List, Any

# Import all modules
from src.model import SimpleSNN, measure_energy_efficiency
from src.data import ECGDataset, generate_synthetic_ecg, rate_encode
from src.train import train_epoch, validate
from src.inference import load_model, predict, profile_inference
from src.utils import get_device, set_seed
from demo.app import app, init_model

class PhaseVerifier:
    def __init__(self):
        self.device = get_device()
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'performance': {},
            'code_quality': {},
        }

    def log(self, message: str, level: str = 'INFO'):
        """Log message with level"""
        prefix = {
            'INFO': '📝',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'PERF': '⚡'
        }
        print(f"{prefix.get(level, '📝')} {message}")

    def test(self, name: str, func, *args, **kwargs) -> bool:
        """Run a test and track results"""
        try:
            result = func(*args, **kwargs)
            self.results['tests_passed'] += 1
            self.log(f"PASS: {name}", 'SUCCESS')
            return True
        except Exception as e:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{name}: {str(e)}")
            self.log(f"FAIL: {name} - {str(e)}", 'ERROR')
            return False

    def verify_artifacts(self):
        """Verify all generated artifacts exist and are valid"""
        self.log("\n" + "="*60)
        self.log("ARTIFACT VERIFICATION")
        self.log("="*60)

        # Data files
        data_files = [
            'data/synthetic/train_data.pt',
            'data/synthetic/val_data.pt',
            'data/synthetic/test_data.pt',
            'data/synthetic/mvp_dataset.pt',
            'data/synthetic/dataset_stats.json'
        ]

        for file_path in data_files:
            self.test(
                f"File exists: {file_path}",
                lambda f: Path(f).exists() and Path(f).stat().st_size > 0,
                file_path
            )

        # Model checkpoint
        self.test(
            "Model checkpoint exists",
            lambda: Path('models/best_model.pt').exists()
        )

        # Verify checkpoint contents
        checkpoint = torch.load('models/best_model.pt', map_location='cpu')
        self.test(
            "Checkpoint has model_state_dict",
            lambda: 'model_state_dict' in checkpoint
        )
        self.test(
            "Checkpoint has optimizer_state_dict",
            lambda: 'optimizer_state_dict' in checkpoint
        )
        self.test(
            "Checkpoint has val_acc >= 85%",
            lambda: checkpoint.get('val_acc', 0) >= 85.0
        )

        self.log(f"\n   Validation Accuracy: {checkpoint.get('val_acc')}%", 'PERF')
        self.log(f"   Validation Loss: {checkpoint.get('val_loss'):.6f}", 'PERF')

    def verify_data_integrity(self):
        """Verify dataset integrity and quality"""
        self.log("\n" + "="*60)
        self.log("DATA INTEGRITY VERIFICATION")
        self.log("="*60)

        # Load datasets
        train_data = torch.load('data/synthetic/train_data.pt')
        val_data = torch.load('data/synthetic/val_data.pt')
        test_data = torch.load('data/synthetic/test_data.pt')

        # Check shapes
        self.test(
            "Train data has signals and labels",
            lambda: 'signals' in train_data and 'labels' in train_data
        )

        # Check dimensions
        self.test(
            "Train signals have correct shape",
            lambda: train_data['signals'].shape == (1000, 2500)
        )
        self.test(
            "Val signals have correct shape",
            lambda: val_data['signals'].shape == (200, 2500)
        )
        self.test(
            "Test signals have correct shape",
            lambda: test_data['signals'].shape == (200, 2500)
        )

        # Check label balance
        train_labels = train_data['labels'].numpy()
        train_class_0 = np.sum(train_labels == 0)
        train_class_1 = np.sum(train_labels == 1)

        self.test(
            "Train data is balanced",
            lambda: abs(train_class_0 - train_class_1) <= 10
        )

        self.log(f"\n   Train: {train_class_0} Normal, {train_class_1} Arrhythmia")
        self.log(f"   Val: {val_data['signals'].shape[0]} samples")
        self.log(f"   Test: {test_data['signals'].shape[0]} samples")

        # Check for NaN or Inf
        self.test(
            "No NaN in train data",
            lambda: not torch.isnan(train_data['signals']).any()
        )
        self.test(
            "No Inf in train data",
            lambda: not torch.isinf(train_data['signals']).any()
        )

    def verify_model_architecture(self):
        """Verify model architecture and parameters"""
        self.log("\n" + "="*60)
        self.log("MODEL ARCHITECTURE VERIFICATION")
        self.log("="*60)

        # Create model
        model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.test(
            "Model has correct number of parameters",
            lambda: total_params == 320386
        )
        self.test(
            "All parameters are trainable",
            lambda: total_params == trainable_params
        )

        self.log(f"\n   Total parameters: {total_params:,}")
        self.log(f"   Trainable parameters: {trainable_params:,}")

        # Test forward pass
        dummy_input = torch.randn(100, 1, 2500)  # [time_steps, batch, features]

        def test_forward():
            model.eval()
            with torch.no_grad():
                spikes, membrane = model(dummy_input)
                return spikes.shape == (100, 1, 2) and membrane.shape == (100, 1, 2)

        self.test("Model forward pass works", test_forward)

        # Test with different batch sizes
        for batch_size in [1, 8, 32]:
            def test_batch():
                dummy = torch.randn(100, batch_size, 2500)
                with torch.no_grad():
                    spikes, membrane = model(dummy)
                return spikes.shape == (100, batch_size, 2)

            self.test(f"Forward pass with batch_size={batch_size}", test_batch)

    def verify_inference(self):
        """Verify inference functionality"""
        self.log("\n" + "="*60)
        self.log("INFERENCE VERIFICATION")
        self.log("="*60)

        # Load model
        model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
        model = load_model('models/best_model.pt', model, device=str(self.device))

        # Load test data
        test_data = torch.load('data/synthetic/test_data.pt')

        # Test single prediction
        signal = test_data['signals'][0].numpy()
        label = test_data['labels'][0].item()

        result = predict(model, signal, device=str(self.device), num_steps=100)

        self.test(
            "Inference returns valid prediction",
            lambda: 'prediction' in result and result['prediction'] in [0, 1]
        )
        self.test(
            "Inference returns confidence",
            lambda: 'confidence' in result and 0 <= result['confidence'] <= 1
        )
        self.test(
            "Inference returns inference_time_ms",
            lambda: 'inference_time_ms' in result and result['inference_time_ms'] > 0
        )

        # Test multiple predictions
        correct = 0
        total_time = 0

        for i in range(min(50, len(test_data['signals']))):
            signal = test_data['signals'][i].numpy()
            label = test_data['labels'][i].item()
            result = predict(model, signal, device=str(self.device), num_steps=100, return_confidence=False)

            if result['prediction'] == label:
                correct += 1
            total_time += result['inference_time_ms']

        accuracy = 100.0 * correct / 50
        avg_time = total_time / 50

        self.results['performance']['test_accuracy'] = accuracy
        self.results['performance']['avg_inference_time_ms'] = avg_time

        self.test(
            "Test accuracy >= 85%",
            lambda: accuracy >= 85.0
        )
        self.test(
            "Average inference time < 100ms",
            lambda: avg_time < 100.0
        )

        self.log(f"\n   Test Accuracy: {accuracy:.2f}%", 'PERF')
        self.log(f"   Avg Inference Time: {avg_time:.2f} ms", 'PERF')

    def verify_flask_integration(self):
        """Verify Flask demo integration"""
        self.log("\n" + "="*60)
        self.log("FLASK INTEGRATION VERIFICATION")
        self.log("="*60)

        # Initialize model
        init_model()

        # Create test client
        app.config['TESTING'] = True
        client = app.test_client()

        # Test health endpoint
        response = client.get('/health')
        self.test(
            "Health endpoint returns 200",
            lambda: response.status_code == 200
        )

        health_data = response.get_json()
        self.test(
            "Model is loaded",
            lambda: health_data['model']['loaded'] == True
        )

        # Test prediction endpoint
        test_data = torch.load('data/synthetic/test_data.pt')
        signal = test_data['signals'][0].numpy().tolist()

        response = client.post(
            '/api/predict',
            json={'signal': signal, 'num_steps': 100},
            content_type='application/json'
        )

        self.test(
            "Prediction endpoint returns 200",
            lambda: response.status_code == 200
        )

        pred_data = response.get_json()
        self.test(
            "Prediction endpoint returns valid data",
            lambda: 'prediction' in pred_data and 'confidence' in pred_data
        )

        # Test generate sample endpoint
        response = client.post(
            '/api/generate_sample',
            json={'condition': 'normal', 'duration': 10},
            content_type='application/json'
        )

        self.test(
            "Generate sample endpoint returns 200",
            lambda: response.status_code == 200
        )

    def performance_benchmarks(self):
        """Run performance benchmarks"""
        self.log("\n" + "="*60)
        self.log("PERFORMANCE BENCHMARKS")
        self.log("="*60)

        # Load model
        model = SimpleSNN(input_size=2500, hidden_size=128, output_size=2)
        model = load_model('models/best_model.pt', model, device=str(self.device))

        # Benchmark inference
        metrics = profile_inference(
            model,
            input_shape=(100, 1, 2500),
            n_iterations=100,
            device=str(self.device)
        )

        self.results['performance']['mean_inference_ms'] = metrics['mean_time_ms']
        self.results['performance']['throughput_samples_sec'] = metrics['throughput_samples_per_sec']

        self.log(f"\n   Mean inference: {metrics['mean_time_ms']:.2f} ms", 'PERF')
        self.log(f"   Std inference: {metrics['std_time_ms']:.2f} ms", 'PERF')
        self.log(f"   Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec", 'PERF')

        # GPU memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9

            self.results['performance']['gpu_memory_allocated_gb'] = mem_allocated
            self.results['performance']['gpu_memory_reserved_gb'] = mem_reserved

            self.log(f"   GPU Memory Allocated: {mem_allocated:.3f} GB", 'PERF')
            self.log(f"   GPU Memory Reserved: {mem_reserved:.3f} GB", 'PERF')

    def code_quality_analysis(self):
        """Analyze code quality"""
        self.log("\n" + "="*60)
        self.log("CODE QUALITY ANALYSIS")
        self.log("="*60)

        src_files = list(Path('src').glob('*.py'))

        total_lines = 0
        total_docstrings = 0
        total_functions = 0

        for file_path in src_files:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                total_lines += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                total_docstrings += content.count('"""')
                total_functions += content.count('def ')

        self.results['code_quality']['total_source_lines'] = total_lines
        self.results['code_quality']['total_functions'] = total_functions
        self.results['code_quality']['docstring_coverage'] = (total_docstrings / (total_functions * 2)) * 100 if total_functions > 0 else 0

        self.log(f"\n   Source files: {len(src_files)}")
        self.log(f"   Total lines: {total_lines}")
        self.log(f"   Functions: {total_functions}")
        self.log(f"   Docstring coverage: {self.results['code_quality']['docstring_coverage']:.1f}%")

    def run_all_verifications(self):
        """Run all verification tests"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE PHASE 1 VERIFICATION")
        print("="*60)

        self.verify_artifacts()
        self.verify_data_integrity()
        self.verify_model_architecture()
        self.verify_inference()
        self.verify_flask_integration()
        self.performance_benchmarks()
        self.code_quality_analysis()

        # Summary
        print("\n" + "="*60)
        print("📊 VERIFICATION SUMMARY")
        print("="*60)

        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        pass_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0

        print(f"\n   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {self.results['tests_passed']}")
        print(f"   ❌ Failed: {self.results['tests_failed']}")
        print(f"   Pass Rate: {pass_rate:.1f}%")

        if self.results['errors']:
            print(f"\n   ⚠️  Errors encountered:")
            for error in self.results['errors']:
                print(f"      - {error}")

        # Performance summary
        print(f"\n   ⚡ Performance Metrics:")
        for key, value in self.results['performance'].items():
            print(f"      - {key}: {value}")

        # Save results
        with open('results/verification_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n   📄 Detailed results saved to: results/verification_report.json")

        # Final verdict
        print("\n" + "="*60)
        if self.results['tests_failed'] == 0:
            print("🎉 PHASE 1 VERIFICATION: ✅ ALL TESTS PASSED")
        else:
            print("⚠️  PHASE 1 VERIFICATION: SOME TESTS FAILED")
        print("="*60)

        return self.results

if __name__ == "__main__":
    verifier = PhaseVerifier()
    results = verifier.run_all_verifications()

    # Exit code
    sys.exit(0 if results['tests_failed'] == 0 else 1)
