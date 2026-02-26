from matplotlib.pylab import power
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

    def forward(self, predictions, targets, mask=None):
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
        # flatten spatial dimensions to compute loss per pixel
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        if mask is not None:
            mask = mask.view(-1)
        else:
            mask = torch.ones_like(predictions)  # If no mask provided, consider all pixels valid

        p = self.power

        # Tweedie loss formula -- following the definition
        term1 = -targets * (predictions ** (1 - p)) / (1 - p)
        term2 = (predictions ** (2 - p)) / (2 - p)

        loss = term1 + term2
        loss = loss * mask

        if self.reduction == 'mean':
            # Average loss across all pixels
            return loss.sum() / mask.sum()  # Normalize by number of valid pixels
        elif self.reduction == 'sum':
            # Total loss (sum of all pixels)
            return loss.sum()
        else:  # 'none'
            # Return loss for each pixel separately
            return loss

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
        

    def forward(self, predictions, targets, mask=None):
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
                
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        if mask is not None:
            mask = mask.view(-1)
        else:
            mask = torch.ones_like(predictions)  # If no mask provided, consider all pixels valid
        # Calculate raw squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply mask: zeroes out the error in all land/deep-ocean pixels
        masked_error = squared_error * mask
        
        # Calculate the mean only over the valid pixels
        loss = masked_error.sum() / mask.sum()
        
        return loss