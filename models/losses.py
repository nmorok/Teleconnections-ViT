import torch
import torch.nn as nn


class TweedieLoss(nn.Module):
    """
    Tweedie loss for modeling non-negative continuous data with excess zeros.
    """
    def __init__(self, power=1.5):
        """
        Initialize Tweedie loss.
        
        Args:
            power (float): Tweedie power parameter (1 < p < 2)
            reduction (str): How to reduce loss across batch
                           'mean' = average, 'sum' = total, 'none' = per-sample
        """
        super().__init__()
        self.power = power

    def forward(self, predictions, targets, mask=None, sample_mask=None):
        """
        Calculate Tweedie loss.
        
        Args:
            predictions: Model predictions [B, 1, H, W] or [B, H, W]
            targets: True densities [B, 1, H, W] or [B, H, W]
        
        Returns:
            loss: Scalar loss value (if reduction='mean' or 'sum')
                  or tensor of losses (if reduction='none')
        """
        
        # Add small epsilon to avoid log(0) and division by zero
        epsilon = 1e-8
        predictions = torch.clamp(predictions, min=epsilon) # clamp forces all values to be at least epsilon
        if mask is not None:
            if mask.dim() == 3 and predictions.dim() == 4:
                mask = mask.unsqueeze(1)
                
        # Flatten spatial dimensions but KEEP the batch dimension: [B, H*W]
        predictions = predictions.view(predictions.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        if mask is not None:
            mask = mask.view(mask.shape[0], -1)
        else:
            mask = torch.ones_like(predictions)  # If no mask provided, consider all pixels valid

        p = self.power

        # Tweedie loss formula -- following the definition
        term1 = -targets * (predictions ** (1 - p)) / (1 - p)
        term2 = (predictions ** (2 - p)) / (2 - p)

        loss = term1 + term2
        

        loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Apply sample mask (zero out 2020)
        if sample_mask is not None:
            loss_per_sample = loss_per_sample * sample_mask
            return loss_per_sample.sum() / sample_mask.sum().clamp(min=1)
        else:
            return loss_per_sample.mean()

class MSELoss_cm(nn.Module):
    """
    MSE loss with spatial masking support.
    """
    def __init__(self):
        """
        Initialize MSE loss.
        
        Args:
            power (float): Tweedie power parameter (1 < p < 2)
            reduction (str): How to reduce loss across batch
                           'mean' = average, 'sum' = total, 'none' = per-sample
        """
        super().__init__()
        

    def forward(self, predictions, targets, mask=None, sample_mask=None):
        """
        Calculate MSE loss.
        
        Args:
            predictions: Model predictions [B, 1, H, W] or [B, H, W]
            targets: True densities [B, 1, H, W] or [B, H, W]
        
        Returns:
            loss: Scalar loss value (if reduction='mean' or 'sum')
                  or tensor of losses (if reduction='none')
        """
        if mask is not None:
            if mask.dim() == 3 and predictions.dim() == 4:
                mask = mask.unsqueeze(1)

        # Flatten spatial dimensions but KEEP the batch dimension: [B, H*W]
        predictions = predictions.view(predictions.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        if mask is not None:
            mask = mask.view(mask.shape[0], -1)
        else:
            mask = torch.ones_like(predictions)  # If no mask provided, consider all pixels valid
        # Calculate raw squared error
        squared_error = (predictions - targets) ** 2
        
        # Per-sample loss: [B]
        loss_per_sample = (squared_error * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if sample_mask is not None:
            loss_per_sample = loss_per_sample * sample_mask
            return loss_per_sample.sum() / sample_mask.sum().clamp(min=1)
        else:
            return loss_per_sample.mean()