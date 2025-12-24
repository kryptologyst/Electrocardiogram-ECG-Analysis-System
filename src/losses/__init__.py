"""Loss functions for ECG analysis."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss is designed to address the one-stage object detection scenario
    where there is an extreme imbalance between foreground and background classes.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Computed focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = -ce_loss
            logpt = logpt * at
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss.
    
    Label smoothing is a regularization technique that prevents the model
    from becoming too confident about its predictions.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """Initialize Label Smoothing Cross Entropy.
        
        Args:
            smoothing: Label smoothing factor.
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Computed loss.
        """
        log_preds = F.log_softmax(inputs, dim=1)
        nll_loss = -log_preds.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks.
    
    Dice Loss is commonly used in medical image segmentation
    to handle class imbalance.
    """
    
    def __init__(self, smooth: float = 1.0):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            inputs: Predicted probabilities.
            targets: Ground truth labels.
            
        Returns:
            Computed Dice loss.
        """
        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_one_hot = targets_one_hot.transpose(1, -1).transpose(2, -1)
        
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Compute Dice coefficient for each class
        dice_scores = []
        for i in range(num_classes):
            input_i = inputs[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (input_i * target_i).sum()
            union = input_i.sum() + target_i.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return average Dice loss
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined loss function for ECG analysis.
    
    Combines multiple loss functions to address different aspects
    of ECG signal classification.
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        smoothing: float = 0.1,
        gamma: float = 2.0,
    ):
        """Initialize Combined Loss.
        
        Args:
            ce_weight: Weight for cross entropy loss.
            focal_weight: Weight for focal loss.
            dice_weight: Weight for dice loss.
            smoothing: Label smoothing factor.
            gamma: Focal loss gamma parameter.
        """
        super().__init__()
        
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing)
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Combined loss value.
        """
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = (
            self.ce_weight * ce +
            self.focal_weight * focal +
            self.dice_weight * dice
        )
        
        return total_loss


class UncertaintyLoss(nn.Module):
    """Loss function that incorporates uncertainty estimation.
    
    This loss encourages the model to be uncertain about its predictions
    when the input is ambiguous or out-of-distribution.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        uncertainty_weight: float = 0.1,
    ):
        """Initialize Uncertainty Loss.
        
        Args:
            base_loss: Base loss function.
            uncertainty_weight: Weight for uncertainty term.
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute uncertainty-aware loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            uncertainty: Uncertainty estimates.
            
        Returns:
            Uncertainty-aware loss.
        """
        base_loss = self.base_loss(inputs, targets)
        
        if uncertainty is not None:
            uncertainty_penalty = uncertainty.mean()
            total_loss = base_loss + self.uncertainty_weight * uncertainty_penalty
        else:
            total_loss = base_loss
        
        return total_loss


def get_loss_function(
    loss_name: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_name: Name of the loss function.
        num_classes: Number of classes.
        class_weights: Optional class weights.
        **kwargs: Additional arguments for loss function.
        
    Returns:
        Loss function instance.
    """
    if loss_name == 'cross_entropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    
    elif loss_name == 'focal':
        alpha = class_weights if class_weights is not None else None
        return FocalLoss(alpha=alpha, **kwargs)
    
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    
    elif loss_name == 'combined':
        return CombinedLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
