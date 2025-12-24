"""Evaluation utilities for ECG analysis models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, Saliency, GradientShap
from captum.attr import visualization as viz

from src.metrics import ECGEvaluator, ClinicalMetrics
from src.utils import get_device


class ECGExplainer:
    """Explainability utilities for ECG models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize ECG explainer.
        
        Args:
            model: Trained ECG model.
            device: Device to run explanations on.
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.saliency = Saliency(self.model)
        self.gradient_shap = GradientShap(self.model)
    
    def explain_prediction(
        self,
        signal: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = 'integrated_gradients',
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Explain model prediction for a single ECG signal.
        
        Args:
            signal: ECG signal to explain.
            target_class: Target class for explanation.
            method: Attribution method to use.
            
        Returns:
            Tuple of (attributions, metadata).
        """
        signal = signal.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                logits = self.model(signal)
                target_class = torch.argmax(logits, dim=1).item()
        
        # Compute attributions
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients.attribute(
                signal, target=target_class, n_steps=50
            )
        elif method == 'saliency':
            attributions = self.saliency.attribute(signal, target=target_class)
        elif method == 'gradient_shap':
            baseline = torch.zeros_like(signal)
            attributions = self.gradient_shap.attribute(
                signal, baselines=baseline, target=target_class
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Get prediction confidence
        with torch.no_grad():
            logits = self.model(signal)
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0, target_class].item()
        
        metadata = {
            'target_class': target_class,
            'confidence': confidence,
            'method': method,
            'probabilities': probabilities[0].cpu().numpy()
        }
        
        return attributions[0].cpu().numpy(), metadata
    
    def plot_explanation(
        self,
        signal: torch.Tensor,
        attributions: np.ndarray,
        metadata: Dict[str, Any],
        class_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot ECG signal with attribution overlay.
        
        Args:
            signal: Original ECG signal.
            attributions: Attribution values.
            metadata: Explanation metadata.
            class_names: Names of classes.
            save_path: Optional path to save the plot.
        """
        signal_np = signal[0].cpu().numpy()
        time_axis = np.arange(len(signal_np)) / 250  # Assuming 250 Hz sampling rate
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot original signal
        ax1.plot(time_axis, signal_np, 'b-', linewidth=1, label='ECG Signal')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title(f'ECG Signal - Predicted: {class_names[metadata["target_class"]]} '
                     f'(Confidence: {metadata["confidence"]:.3f})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot attributions
        ax2.plot(time_axis, attributions[0], 'r-', linewidth=1, label='Attributions')
        ax2.fill_between(time_axis, attributions[0], alpha=0.3, color='red')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Attribution')
        ax2.set_title(f'Attribution Map ({metadata["method"]})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def batch_explain(
        self,
        dataloader: DataLoader,
        num_samples: int = 10,
        method: str = 'integrated_gradients',
        class_names: List[str] = None,
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Explain predictions for a batch of samples.
        
        Args:
            dataloader: Data loader.
            num_samples: Number of samples to explain.
            method: Attribution method.
            class_names: Names of classes.
            
        Returns:
            List of (attributions, metadata) tuples.
        """
        explanations = []
        count = 0
        
        for signals, labels in dataloader:
            if count >= num_samples:
                break
                
            for signal, label in zip(signals, labels):
                if count >= num_samples:
                    break
                    
                attributions, metadata = self.explain_prediction(
                    signal.unsqueeze(0), method=method
                )
                metadata['true_label'] = label.item()
                explanations.append((attributions, metadata))
                count += 1
        
        return explanations


class UncertaintyEstimator:
    """Uncertainty estimation for ECG models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize uncertainty estimator.
        
        Args:
            model: Trained ECG model.
            device: Device to run on.
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def monte_carlo_dropout(
        self,
        signal: torch.Tensor,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            signal: ECG signal.
            num_samples: Number of Monte Carlo samples.
            
        Returns:
            Tuple of (mean_predictions, uncertainty).
        """
        signal = signal.to(self.device)
        
        # Enable dropout during inference
        self.model.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(signal)
                probabilities = torch.softmax(logits, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        # Disable dropout
        self.model.eval()
        
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        
        # Compute uncertainty as entropy
        uncertainty = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=1)
        
        return mean_predictions, uncertainty
    
    def ensemble_uncertainty(
        self,
        models: List[nn.Module],
        signal: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty using model ensemble.
        
        Args:
            models: List of trained models.
            signal: ECG signal.
            
        Returns:
            Tuple of (mean_predictions, uncertainty).
        """
        signal = signal.to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for model in models:
                model.eval()
                logits = model(signal)
                probabilities = torch.softmax(logits, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        
        # Compute uncertainty as variance
        uncertainty = np.var(predictions, axis=0).sum(axis=1)
        
        return mean_predictions, uncertainty


class ECGEvaluatorExtended(ECGEvaluator):
    """Extended ECG evaluator with additional analysis capabilities."""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """Initialize extended ECG evaluator."""
        super().__init__(num_classes, class_names)
        self.explainer = None
        self.uncertainty_estimator = None
    
    def set_explainer(self, model: nn.Module, device: torch.device) -> None:
        """Set explainer for the evaluator."""
        self.explainer = ECGExplainer(model, device)
    
    def set_uncertainty_estimator(self, model: nn.Module, device: torch.device) -> None:
        """Set uncertainty estimator for the evaluator."""
        self.uncertainty_estimator = UncertaintyEstimator(model, device)
    
    def comprehensive_evaluation(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str = 'outputs',
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            model: Trained model.
            test_loader: Test data loader.
            device: Device to run on.
            output_dir: Output directory for results.
            
        Returns:
            Dictionary of evaluation results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set explainer and uncertainty estimator
        self.set_explainer(model, device)
        self.set_uncertainty_estimator(model, device)
        
        # Standard evaluation
        metrics, (y_true, y_pred, y_prob) = self.evaluate_model(
            model, test_loader, device, return_predictions=True
        )
        
        # Generate plots
        self.metrics.plot_confusion_matrix(y_true, y_pred, 
                                         save_path=str(output_path / 'confusion_matrix.png'))
        
        if y_prob is not None:
            self.metrics.plot_calibration_curve(y_true, y_prob,
                                              save_path=str(output_path / 'calibration_curve.png'))
        
        # Generate explanation plots
        explanations = self.explainer.batch_explain(
            test_loader, num_samples=5, class_names=self.class_names
        )
        
        for i, (attributions, metadata) in enumerate(explanations):
            # Get original signal
            signal_idx = i
            for signals, _ in test_loader:
                if signal_idx < len(signals):
                    signal = signals[signal_idx:signal_idx+1]
                    break
                signal_idx -= len(signals)
            
            self.explainer.plot_explanation(
                signal, attributions, metadata, self.class_names,
                save_path=str(output_path / f'explanation_{i}.png')
            )
        
        # Generate comprehensive report
        report = self.generate_report(y_true, y_pred, y_prob)
        
        # Save results
        results = {
            'metrics': metrics,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            },
            'explanations': explanations,
            'report': report
        }
        
        # Save results to file
        import json
        with open(output_path / 'evaluation_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        return results


def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Comprehensive model evaluation pipeline.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        config: Evaluation configuration.
        device: Device to run on.
        
    Returns:
        Dictionary of evaluation results.
    """
    if device is None:
        device = get_device()
    
    evaluator = ECGEvaluatorExtended(
        num_classes=config.get('num_classes', 4),
        class_names=config.get('class_names', None)
    )
    
    return evaluator.comprehensive_evaluation(
        model, test_loader, device, 
        config.get('output_dir', 'outputs')
    )
