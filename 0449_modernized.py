#!/usr/bin/env python3
"""
Project 449: ECG Analysis System - Modernized Version

This is a modernized, production-ready version of the original ECG analysis system.
It demonstrates advanced deep learning models, clinical evaluation metrics, and
explainable AI capabilities for ECG signal classification.

IMPORTANT DISCLAIMER:
This software is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
NOT FOR CLINICAL USE - This system is not intended for medical diagnosis,
clinical decision making, or patient care.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

# Import our modernized modules
from src.utils import set_seed, get_device, count_parameters
from src.data import ECGDataset, ECGSignalProcessor
from src.models import ECGCNN, ECGResNet, ECGTransformer
from src.losses import get_loss_function
from src.metrics import ClinicalMetrics
from src.train import create_trainer


def print_banner():
    """Print system banner with disclaimer."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    ECG ANALYSIS SYSTEM                        ║
    ║                  (Modernized Version)                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  RESEARCH AND EDUCATIONAL PURPOSES ONLY                      ║
    ║  NOT FOR CLINICAL USE - NO MEDICAL ADVICE                   ║
    ║  Always consult qualified healthcare professionals          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_config() -> Dict[str, Any]:
    """Load configuration for the demo."""
    config = {
        'model': {
            'name': 'ECGResNet',
            'input_length': 250,
            'num_classes': 4,
            'dropout_rate': 0.5
        },
        'data': {
            'num_samples': 1000,
            'seq_length': 250,
            'batch_size': 32,
            'num_workers': 2,
            'class_names': ['Normal', 'Atrial Fibrillation', 'Ventricular Tachycardia', 'Other Arrhythmia']
        },
        'training': {
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'loss_function': 'combined',
            'loss_kwargs': {
                'ce_weight': 1.0,
                'focal_weight': 0.5,
                'dice_weight': 0.3,
                'smoothing': 0.1,
                'gamma': 2.0
            }
        },
        'seed': 42
    }
    return config


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create ECG model based on configuration."""
    model_config = config['model']
    model_name = model_config['name']
    
    print(f"Creating {model_name} model...")
    
    if model_name == 'ECGCNN':
        model = ECGCNN(
            input_length=model_config['input_length'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate']
        )
    elif model_name == 'ECGResNet':
        model = ECGResNet(
            input_length=model_config['input_length'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate']
        )
    elif model_name == 'ECGTransformer':
        model = ECGTransformer(
            input_length=model_config['input_length'],
            num_classes=model_config['num_classes'],
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout_rate=model_config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    data_config = config['data']
    
    print("Creating ECG dataset...")
    
    # Create preprocessor
    preprocessor = ECGSignalProcessor(
        sampling_rate=250,
        filter_low=0.5,
        filter_high=40.0,
        normalize=True
    )
    
    # Create dataset
    dataset = ECGDataset(
        num_samples=data_config['num_samples'],
        seq_length=data_config['seq_length'],
        num_classes=config['model']['num_classes'],
        synthetic=True,  # Use synthetic data for demo
        preprocessor=preprocessor
    )
    
    # Split dataset (70% train, 30% validation)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the ECG model."""
    print("\n" + "="*60)
    print("TRAINING ECG ANALYSIS MODEL")
    print("="*60)
    
    device = get_device()
    model = model.to(device)
    
    # Create trainer
    trainer = create_trainer(model, train_loader, val_loader, config, device)
    
    # Train model
    results = trainer.train(config['training']['num_epochs'])
    
    return results


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*60)
    
    device = get_device()
    model = model.to(device)
    
    # Create evaluator
    evaluator = ClinicalMetrics(
        num_classes=config['model']['num_classes'],
        class_names=config['data']['class_names']
    )
    
    # Evaluate model
    metrics = evaluator.evaluate_model(model, val_loader, device)
    
    return metrics


def demonstrate_explainability(model: nn.Module, val_loader: DataLoader, config: Dict[str, Any]):
    """Demonstrate model explainability features."""
    print("\n" + "="*60)
    print("MODEL EXPLAINABILITY DEMONSTRATION")
    print("="*60)
    
    device = get_device()
    model = model.to(device)
    
    # Get a sample from validation set
    for signals, labels in val_loader:
        sample_signal = signals[0:1].to(device)
        sample_label = labels[0].item()
        break
    
    print(f"Sample ECG signal (True class: {config['data']['class_names'][sample_label]})")
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(sample_signal)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"Predicted class: {config['data']['class_names'][predicted_class]}")
    print(f"Confidence: {confidence:.3f}")
    
    # Show class probabilities
    print("\nClass Probabilities:")
    for i, class_name in enumerate(config['data']['class_names']):
        prob = probabilities[0, i].item()
        print(f"  {class_name}: {prob:.3f}")
    
    # Note: Full explainability would require Captum installation
    print("\nNote: Full attribution analysis requires Captum library.")
    print("Install with: pip install captum")


def plot_sample_signals(dataset: ECGDataset, config: Dict[str, Any]):
    """Plot sample ECG signals from each class."""
    print("\n" + "="*60)
    print("SAMPLE ECG SIGNALS BY CLASS")
    print("="*60)
    
    # Get one sample from each class
    class_samples = {}
    for i in range(len(dataset)):
        signal, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = signal.squeeze().numpy()
        if len(class_samples) == config['model']['num_classes']:
            break
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (class_idx, signal) in enumerate(class_samples.items()):
        time_axis = np.arange(len(signal)) / 250  # 250 Hz sampling rate
        
        axes[i].plot(time_axis, signal, 'b-', linewidth=1)
        axes[i].set_title(f"{config['data']['class_names'][class_idx]} ECG")
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (mV)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_ecg_signals.png', dpi=300, bbox_inches='tight')
    print("Sample ECG signals saved to 'sample_ecg_signals.png'")
    plt.show()


def main():
    """Main demonstration function."""
    print_banner()
    
    # Load configuration
    config = load_config()
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    print(f"Random seed set to: {config['seed']}")
    
    # Create model
    model = create_model(config)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Plot sample signals
    dataset = ECGDataset(
        num_samples=100,
        seq_length=config['data']['seq_length'],
        num_classes=config['model']['num_classes'],
        synthetic=True
    )
    plot_sample_signals(dataset, config)
    
    # Train model
    training_results = train_model(model, train_loader, val_loader, config)
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, config)
    
    # Print evaluation results
    print("\nEVALUATION RESULTS:")
    print("-" * 40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'auc_roc_ovr' in metrics:
        print(f"ROC-AUC (OvR): {metrics['auc_roc_ovr']:.4f}")
    
    if 'brier_score' in metrics:
        print(f"Brier Score: {metrics['brier_score']:.4f}")
    
    if 'ece' in metrics:
        print(f"Expected Calibration Error: {metrics['ece']:.4f}")
    
    # Demonstrate explainability
    demonstrate_explainability(model, val_loader, config)
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print("This modernized ECG analysis system demonstrates:")
    print("✓ Advanced deep learning architectures (CNN, ResNet, Transformer)")
    print("✓ Clinical evaluation metrics and calibration")
    print("✓ Synthetic ECG data generation")
    print("✓ Model explainability and uncertainty estimation")
    print("✓ Production-ready code structure and documentation")
    print("\nFor interactive demos, run:")
    print("  streamlit run demo/streamlit_app.py")
    print("\nFor comprehensive evaluation, run:")
    print("  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("\nRemember: This system is for RESEARCH AND EDUCATIONAL PURPOSES ONLY")
    print("NOT FOR CLINICAL USE - Always consult healthcare professionals")


if __name__ == "__main__":
    main()
