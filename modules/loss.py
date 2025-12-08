import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for binary segmentation.
    
    Args:
        pred: Model predictions (logits, before sigmoid)
        target: Ground truth masks (0-1 range)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5):
    """
    Combined BCE + Dice loss.
    
    Args:
        pred: Model predictions (logits, before sigmoid)
        target: Ground truth masks (0-1 range)
        bce_weight: Weight for BCE component (0.0-1.0)
                   Dice weight = 1 - bce_weight
    
    Returns:
        Combined loss value
    
    Example:
        loss = combined_loss(outputs, masks, bce_weight=0.5)
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice


# ============================================================================
# DEPRECATED: Old class-based version (kept for backward compatibility)
# ============================================================================

class BCEDiceLoss(nn.Module):
    """
    DEPRECATED: Use combined_loss() function instead.
    
    Combination of BCEWithLogitsLoss and Dice Loss.
    This class is kept for backward compatibility only.
    
    Example (old way):
        criterion = BCEDiceLoss(bce_weight=0.5)
        loss = criterion(outputs, masks)
    
    Example (new way - recommended):
        loss = combined_loss(outputs, masks, bce_weight=0.5)
    """
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        print("WARNING: BCEDiceLoss class is deprecated. Use combined_loss() function instead.")
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        return combined_loss(inputs, targets, bce_weight=self.bce_weight)