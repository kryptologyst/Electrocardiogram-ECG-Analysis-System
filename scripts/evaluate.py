#!/usr/bin/env python3
"""Comprehensive evaluation script for ECG analysis system."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import get_device, load_checkpoint
from src.data import ECGDataset, ECGSignalProcessor
from src.models import ECGCNN, ECGResNet, ECGTransformer
from src.eval import evaluate_model_comprehensive


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_name = model_config['name']
    
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
        transformer_config = model_config['transformer']
        model = ECGTransformer(
            input_length=model_config['input_length'],
            num_classes=model_config['num_classes'],
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            num_layers=transformer_config['num_layers'],
            dropout_rate=model_config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def create_test_loader(config: Dict[str, Any]) -> torch.utils.data.DataLoader:
    """Create test data loader."""
    data_config = config['data']
    
    # Create preprocessor
    preprocessor = ECGSignalProcessor(
        sampling_rate=data_config['preprocessing']['sampling_rate'],
        filter_low=data_config['preprocessing']['filter_low'],
        filter_high=data_config['preprocessing']['filter_high'],
        normalize=data_config['preprocessing']['normalize']
    )
    
    # Create test dataset
    test_dataset = ECGDataset(
        num_samples=data_config['num_samples'] // 4,  # Smaller test set
        seq_length=data_config['seq_length'],
        num_classes=config['model']['num_classes'],
        synthetic=data_config['synthetic'],
        preprocessor=preprocessor
    )
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return test_loader


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ECG analysis model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Created model: {config['model']['name']}")
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create test loader
    test_loader = create_test_loader(config)
    print(f"Created test loader with {len(test_loader)} batches")
    
    # Perform comprehensive evaluation
    print("Starting comprehensive evaluation...")
    results = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Print summary results
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    metrics = results['metrics']
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'auc_roc_ovr' in metrics:
        print(f"ROC-AUC (OvR): {metrics['auc_roc_ovr']:.4f}")
    if 'auc_roc_ovo' in metrics:
        print(f"ROC-AUC (OvO): {metrics['auc_roc_ovo']:.4f}")
    
    if 'brier_score' in metrics:
        print(f"Brier Score: {metrics['brier_score']:.4f}")
    if 'ece' in metrics:
        print(f"Expected Calibration Error: {metrics['ece']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
