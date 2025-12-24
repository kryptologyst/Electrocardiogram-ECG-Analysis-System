#!/usr/bin/env python3
"""Main training script for ECG analysis system."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torch.utils.data import DataLoader, random_split
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device
from src.data import ECGDataset, ECGSignalProcessor
from src.models import ECGCNN, ECGResNet, ECGTransformer
from src.train import create_trainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Created model.
    """
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


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Create data loaders for training and validation.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_config = config['data']
    
    # Create preprocessor
    preprocessor = ECGSignalProcessor(
        sampling_rate=data_config['preprocessing']['sampling_rate'],
        filter_low=data_config['preprocessing']['filter_low'],
        filter_high=data_config['preprocessing']['filter_high'],
        normalize=data_config['preprocessing']['normalize']
    )
    
    # Create dataset
    dataset = ECGDataset(
        num_samples=data_config['num_samples'],
        seq_length=data_config['seq_length'],
        num_classes=config['model']['num_classes'],
        synthetic=data_config['synthetic'],
        preprocessor=preprocessor
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ECG analysis model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate model, do not train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Created model: {config['model']['name']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Create trainer
    trainer = create_trainer(model, train_loader, val_loader, config, device)
    
    if args.eval_only:
        # Load checkpoint if specified
        if args.resume:
            from src.utils import load_checkpoint
            load_checkpoint(args.resume, model, device=device)
            print(f"Loaded checkpoint from {args.resume}")
        
        # Evaluate model
        test_metrics = trainer.evaluate(test_loader)
        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        # Train model
        training_results = trainer.train(config['training']['num_epochs'])
        
        print("\nTraining completed!")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print("Best validation metrics:")
        for key, value in training_results['best_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
