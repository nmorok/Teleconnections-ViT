import torch
import torch.nn as nn


class TweedieLoss(nn.Module):
    """
    Tweedie loss for modeling non-negative continuous data with excess zeros.
    """
    def __init__(self, power=1.5, reduction='mean'):
        """
        Initialize Tweedie loss.
        
        Args:
            power (float): Tweedie power parameter (1 < p < 2)
            reduction (str): How to reduce loss across batch
                           'mean' = average, 'sum' = total, 'none' = per-sample
        """
        super().__init__()
        self.power = power
        self.reduction = reduction

    def forward(self, predictions, targets):
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

        # flatten spatial dimensions to compute loss per pixel
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        p = self.power

        # Tweedie loss formula -- following the definition
        term1 = -targets * (predictions ** (1 - p)) / (1 - p)
        term2 = (predictions ** (2 - p)) / (2 - p)

        loss = term1 + term2

        if self.reduction == 'mean':
            # Average loss across all pixels
            return loss.mean()
        elif self.reduction == 'sum':
            # Total loss (sum of all pixels)
            return loss.sum()
        else:  # 'none'
            # Return loss for each pixel separately
            return loss

