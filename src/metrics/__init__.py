"""Clinical evaluation metrics for ECG analysis."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


class ClinicalMetrics:
    """Clinical evaluation metrics for ECG analysis."""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """Initialize clinical metrics.
        
        Args:
            num_classes: Number of classes.
            class_names: Names of classes.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive clinical metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary of computed metrics.
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision_per_class[i]
            metrics[f'{class_name}_recall'] = recall_per_class[i]
            metrics[f'{class_name}_f1'] = f1_per_class[i]
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            if self.num_classes == 2:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                try:
                    metrics['auc_roc_ovr'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
                    metrics['auc_roc_ovo'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovo', average='weighted'
                    )
                except ValueError:
                    # Handle case where some classes are missing
                    metrics['auc_roc_ovr'] = 0.0
                    metrics['auc_roc_ovo'] = 0.0
        
        # Clinical-specific metrics
        clinical_metrics = self._compute_clinical_metrics(y_true, y_pred, y_prob)
        metrics.update(clinical_metrics)
        
        return metrics
    
    def _compute_clinical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute clinical-specific metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary of clinical metrics.
        """
        metrics = {}
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Sensitivity (Recall) and Specificity for each class
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics[f'{self.class_names[i]}_sensitivity'] = sensitivity
            metrics[f'{self.class_names[i]}_specificity'] = specificity
            
            # Positive and Negative Predictive Values
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            metrics[f'{self.class_names[i]}_ppv'] = ppv
            metrics[f'{self.class_names[i]}_npv'] = npv
        
        # Overall sensitivity and specificity (macro average)
        sensitivities = [metrics[f'{name}_sensitivity'] for name in self.class_names]
        specificities = [metrics[f'{name}_specificity'] for name in self.class_names]
        
        metrics['macro_sensitivity'] = np.mean(sensitivities)
        metrics['macro_specificity'] = np.mean(specificities)
        
        # Calibration metrics (if probabilities available)
        if y_prob is not None:
            calibration_metrics = self._compute_calibration_metrics(y_true, y_prob)
            metrics.update(calibration_metrics)
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute calibration metrics.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary of calibration metrics.
        """
        metrics = {}
        
        if self.num_classes == 2:
            # Binary calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob[:, 1], n_bins=10
            )
            
            # Brier Score
            brier_score = np.mean((y_prob[:, 1] - y_true) ** 2)
            metrics['brier_score'] = brier_score
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            metrics['ece'] = ece
            
        else:
            # Multi-class calibration
            # Convert to one-hot encoding
            y_true_one_hot = np.eye(self.num_classes)[y_true]
            
            # Brier Score for multi-class
            brier_score = np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))
            metrics['brier_score'] = brier_score
            
            # ECE for multi-class
            ece = 0.0
            for i in range(self.num_classes):
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_one_hot[:, i], y_prob[:, i], n_bins=10
                )
                ece += np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            metrics['ece'] = ece / self.num_classes
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Optional path to save the plot.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            save_path: Optional path to save the plot.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        
        if self.num_classes == 2:
            # Binary calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob[:, 1], n_bins=10
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"Model (ECE = {self._compute_calibration_metrics(y_true, y_prob)['ece']:.3f})")
            
        else:
            # Multi-class calibration
            y_true_one_hot = np.eye(self.num_classes)[y_true]
            
            for i, class_name in enumerate(self.class_names):
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_one_hot[:, i], y_prob[:, i], n_bins=10
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, "o-",
                        label=f"{class_name}")
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ECGEvaluator:
    """Comprehensive evaluator for ECG models."""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """Initialize ECG evaluator.
        
        Args:
            num_classes: Number of classes.
            class_names: Names of classes.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.metrics = ClinicalMetrics(num_classes, class_names)
        
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Evaluate model on dataset.
        
        Args:
            model: Trained model.
            dataloader: Data loader for evaluation.
            device: Device to run evaluation on.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of metrics and optionally predictions.
        """
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                signals, labels = batch
                signals = signals.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(signals)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self.metrics.compute_metrics(y_true, y_pred, y_prob)
        
        if return_predictions:
            return metrics, (y_true, y_pred, y_prob)
        else:
            return metrics
    
    def generate_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            save_path: Optional path to save the report.
            
        Returns:
            Formatted evaluation report.
        """
        metrics = self.metrics.compute_metrics(y_true, y_pred, y_prob)
        
        report = "ECG Analysis Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall performance
        report += "Overall Performance:\n"
        report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"  Precision: {metrics['precision']:.4f}\n"
        report += f"  Recall: {metrics['recall']:.4f}\n"
        report += f"  F1-Score: {metrics['f1_score']:.4f}\n\n"
        
        # Per-class performance
        report += "Per-Class Performance:\n"
        for class_name in self.class_names:
            report += f"  {class_name}:\n"
            report += f"    Precision: {metrics[f'{class_name}_precision']:.4f}\n"
            report += f"    Recall: {metrics[f'{class_name}_recall']:.4f}\n"
            report += f"    F1-Score: {metrics[f'{class_name}_f1']:.4f}\n"
            report += f"    Sensitivity: {metrics[f'{class_name}_sensitivity']:.4f}\n"
            report += f"    Specificity: {metrics[f'{class_name}_specificity']:.4f}\n"
            report += f"    PPV: {metrics[f'{class_name}_ppv']:.4f}\n"
            report += f"    NPV: {metrics[f'{class_name}_npv']:.4f}\n\n"
        
        # Calibration metrics
        if y_prob is not None:
            report += "Calibration Metrics:\n"
            report += f"  Brier Score: {metrics['brier_score']:.4f}\n"
            report += f"  Expected Calibration Error: {metrics['ece']:.4f}\n\n"
        
        # AUC metrics
        if y_prob is not None:
            report += "AUC Metrics:\n"
            if self.num_classes == 2:
                report += f"  ROC-AUC: {metrics['auc_roc']:.4f}\n"
                report += f"  PR-AUC: {metrics['auc_pr']:.4f}\n"
            else:
                report += f"  ROC-AUC (OvR): {metrics['auc_roc_ovr']:.4f}\n"
                report += f"  ROC-AUC (OvO): {metrics['auc_roc_ovo']:.4f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
