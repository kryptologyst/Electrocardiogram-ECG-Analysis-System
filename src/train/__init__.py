"""Training utilities for ECG analysis models."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import yaml

from src.utils import EarlyStopping, save_checkpoint, load_checkpoint
from src.metrics import ECGEvaluator


class ECGTrainer:
    """Trainer class for ECG analysis models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        class_names: Optional[list] = None,
    ):
        """Initialize ECG trainer.
        
        Args:
            model: ECG model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Device to train on.
            config: Training configuration.
            class_names: Names of classes.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.class_names = class_names or [f"Class_{i}" for i in range(config.get('num_classes', 4))]
        
        # Initialize evaluator
        self.evaluator = ECGEvaluator(
            num_classes=config.get('num_classes', 4),
            class_names=self.class_names
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001),
            restore_best_weights=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(config.get('log_dir', 'runs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (signals, labels) in enumerate(pbar):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        # Compute epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        
        # Compute accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        metrics = {
            'train_loss': epoch_loss,
            'train_accuracy': accuracy,
        }
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        running_loss = 0.0
        
        with torch.no_grad():
            for signals, labels in self.val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
        
        epoch_loss = running_loss / len(self.val_loader)
        
        # Compute comprehensive validation metrics
        val_metrics = self.evaluator.evaluate_model(
            self.model, self.val_loader, self.device
        )
        
        val_metrics['val_loss'] = epoch_loss
        
        return val_metrics
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Dictionary containing training history and best metrics.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Check for best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_metrics = val_metrics.copy()
                
                # Save best model
                checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / 'best_model.pth'
                checkpoint_path.parent.mkdir(exist_ok=True)
                
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['val_loss'],
                    val_metrics, str(checkpoint_path)
                )
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss'], self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        final_checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / 'final_model.pth'
        save_checkpoint(
            self.model, self.optimizer, epoch, val_metrics['val_loss'],
            val_metrics, str(final_checkpoint_path)
        )
        
        self.writer.close()
        
        return {
            'history': history,
            'best_metrics': self.best_metrics,
            'best_val_loss': self.best_val_loss,
        }
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to tensorboard.
        
        Args:
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            epoch: Current epoch.
        """
        # Loss metrics
        self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        self.writer.add_scalar('Loss/Validation', val_metrics['val_loss'], epoch)
        
        # Accuracy metrics
        self.writer.add_scalar('Accuracy/Train', train_metrics['train_accuracy'], epoch)
        self.writer.add_scalar('Accuracy/Validation', val_metrics['val_accuracy'], epoch)
        
        # Additional validation metrics
        for key, value in val_metrics.items():
            if key not in ['val_loss', 'val_accuracy']:
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        # Learning rate
        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('Learning_Rate', param_group['lr'], epoch)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary of test metrics.
        """
        print("Evaluating model on test set...")
        
        # Load best model
        checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / 'best_model.pth'
        if checkpoint_path.exists():
            load_checkpoint(str(checkpoint_path), self.model, device=self.device)
            print(f"Loaded best model from {checkpoint_path}")
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(
            self.model, test_loader, self.device
        )
        
        # Generate report
        report = self.evaluator.generate_report(
            *self.evaluator.evaluate_model(
                self.model, test_loader, self.device, return_predictions=True
            )[1]
        )
        
        print("\n" + report)
        
        # Save report
        report_path = Path(self.config.get('output_dir', 'outputs')) / 'evaluation_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return test_metrics


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> ECGTrainer:
    """Create ECG trainer with configuration.
    
    Args:
        model: ECG model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device to train on.
        
    Returns:
        Configured ECG trainer.
    """
    # Loss function
    from src.losses import get_loss_function
    criterion = get_loss_function(
        config.get('loss_function', 'cross_entropy'),
        config.get('num_classes', 4),
        **config.get('loss_kwargs', {})
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Scheduler
    if config.get('scheduler', None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
            verbose=True
        )
    
    return ECGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        class_names=config.get('class_names', None)
    )
