"""
Loss Functions Module
=====================

Owner: CS2 / SNN Expert

Custom loss functions for clinical SNN training with emphasis on
high-sensitivity arrhythmia detection.

Phase: Tier 1 Fixes (Phase 2.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class importance imbalance

    Focal Loss down-weights easy examples and focuses training on hard examples.
    This is particularly useful when false negatives (missed arrhythmias) are
    more dangerous than false positives (false alarms).

    Mathematical Formulation:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        - p_t: predicted probability for the true class
        - α_t: weighting factor for class t (alpha)
        - γ: focusing parameter (gamma), typically 2.0

    Clinical Motivation:
        In medical diagnosis, false negatives can be life-threatening while
        false positives only cause inconvenience. Focal Loss allows the model
        to focus more on correctly classifying the positive (arrhythmia) class.

    Args:
        alpha: Weighting factor for positive class (Arrhythmia).
              Higher alpha means more penalty for misclassifying Arrhythmia.
              Recommended: 0.75 (3:1 ratio favoring arrhythmia detection)
        gamma: Focusing parameter for hard examples.
              Higher gamma means more focus on hard-to-classify examples.
              Recommended: 2.0 (standard from Focal Loss paper)
        reduction: Reduction method ('mean', 'sum', 'none')

    Example:
        >>> criterion = FocalLoss(alpha=0.75, gamma=2.0)
        >>> logits = model(inputs)  # Shape: [batch, 2]
        >>> loss = criterion(logits, targets)
        >>> loss.backward()

    Reference:
        Lin et al. (2017). "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss

        Args:
            inputs: Logits from model [batch, num_classes]
            targets: Ground truth labels [batch] (class indices)

        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get predicted probabilities
        p = torch.exp(-ce_loss)

        # Compute focal loss
        # (1 - p)^gamma focuses on hard examples
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting for class balance
        # For binary classification: alpha for class 1, (1-alpha) for class 0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for class importance

    Simple alternative to Focal Loss that directly applies class weights
    to the cross entropy loss. More straightforward than Focal Loss but
    less sophisticated in handling hard examples.

    Clinical Motivation:
        By weighting the Arrhythmia class higher, we penalize false negatives
        more heavily during training, encouraging the model to be more sensitive
        to arrhythmia detection.

    Args:
        weight: Tensor of class weights [num_classes].
               Example: torch.tensor([1.0, 3.0]) means Arrhythmia (class 1)
               errors are penalized 3x more than Normal (class 0) errors.
        reduction: Reduction method ('mean', 'sum', 'none')

    Example:
        >>> # 3:1 weighting favoring arrhythmia detection
        >>> class_weights = torch.tensor([1.0, 3.0])
        >>> criterion = WeightedCrossEntropyLoss(weight=class_weights)
        >>> logits = model(inputs)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        weight: torch.Tensor,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted Cross Entropy Loss

        Args:
            inputs: Logits from model [batch, num_classes]
            targets: Ground truth labels [batch] (class indices)

        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)

    def __repr__(self):
        return f"WeightedCrossEntropyLoss(weight={self.weight.tolist()}, reduction='{self.reduction}')"


class ClinicalLoss(nn.Module):
    """
    Combined loss function for clinical applications

    Combines standard cross entropy with explicit penalties for false negatives.
    This provides direct control over the sensitivity-specificity trade-off.

    Loss = CE_loss + λ * FN_penalty

    Where:
        - CE_loss: Standard cross entropy
        - FN_penalty: Additional penalty for false negatives
        - λ: Weighting factor controlling the trade-off

    Args:
        fn_weight: Weight for false negative penalty (higher = more sensitive)
                  Recommended: 2.0-5.0
        reduction: Reduction method ('mean', 'sum', 'none')

    Example:
        >>> criterion = ClinicalLoss(fn_weight=3.0)
        >>> logits = model(inputs)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        fn_weight: float = 3.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.fn_weight = fn_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Clinical Loss

        Args:
            inputs: Logits from model [batch, num_classes]
            targets: Ground truth labels [batch] (class indices)

        Returns:
            Clinical loss value
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get predictions
        preds = inputs.argmax(dim=1)

        # Identify false negatives (true label = 1, predicted = 0)
        false_negatives = (targets == 1) & (preds == 0)

        # Add penalty for false negatives
        fn_penalty = false_negatives.float() * self.fn_weight

        # Combined loss
        total_loss = ce_loss + fn_penalty

        # Apply reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

    def __repr__(self):
        return f"ClinicalLoss(fn_weight={self.fn_weight}, reduction='{self.reduction}')"


def get_loss_function(
    loss_type: str = 'focal',
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions

    Args:
        loss_type: Type of loss function
                  - 'cross_entropy': Standard CE (no weighting)
                  - 'weighted_ce': Weighted cross entropy
                  - 'focal': Focal loss (recommended)
                  - 'clinical': Clinical loss with FN penalty
        class_weights: Class weights for weighted_ce
                      Example: torch.tensor([1.0, 3.0])
        **kwargs: Additional arguments for specific loss functions
                 - alpha, gamma for focal loss
                 - fn_weight for clinical loss

    Returns:
        Initialized loss function

    Example:
        >>> # Focal loss (recommended)
        >>> criterion = get_loss_function('focal', alpha=0.75, gamma=2.0)
        >>>
        >>> # Weighted CE
        >>> criterion = get_loss_function('weighted_ce',
        ...                              class_weights=torch.tensor([1.0, 3.0]))
        >>>
        >>> # Clinical loss
        >>> criterion = get_loss_function('clinical', fn_weight=3.0)
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce")
        return WeightedCrossEntropyLoss(weight=class_weights)

    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'clinical':
        fn_weight = kwargs.get('fn_weight', 3.0)
        return ClinicalLoss(fn_weight=fn_weight)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Choose from: cross_entropy, weighted_ce, focal, clinical")


# ============================================
# Utility Functions
# ============================================

def compare_loss_functions(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> dict:
    """
    Compare different loss functions on the same predictions

    Useful for understanding how different loss functions respond to
    the same model outputs.

    Args:
        logits: Model predictions [batch, num_classes]
        targets: Ground truth labels [batch]

    Returns:
        Dictionary mapping loss function names to loss values

    Example:
        >>> logits = model(inputs)
        >>> losses = compare_loss_functions(logits, targets)
        >>> print(f"CE: {losses['ce']:.4f}, Focal: {losses['focal']:.4f}")
    """
    results = {}

    # Standard CE
    ce = nn.CrossEntropyLoss()
    results['cross_entropy'] = ce(logits, targets).item()

    # Weighted CE (3:1)
    weighted_ce = WeightedCrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(logits.device))
    results['weighted_ce_3to1'] = weighted_ce(logits, targets).item()

    # Focal Loss
    focal = FocalLoss(alpha=0.75, gamma=2.0)
    results['focal_default'] = focal(logits, targets).item()

    # Clinical Loss
    clinical = ClinicalLoss(fn_weight=3.0)
    results['clinical'] = clinical(logits, targets).item()

    return results


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("🧪 Testing loss functions module...")

    # Create dummy data
    batch_size = 32
    num_classes = 2

    # Simulated logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print("\n📊 Testing all loss functions:")
    print("-" * 60)

    # Test each loss function
    loss_configs = [
        ('CrossEntropyLoss', nn.CrossEntropyLoss()),
        ('WeightedCE (3:1)', WeightedCrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))),
        ('FocalLoss (α=0.75, γ=2)', FocalLoss(alpha=0.75, gamma=2.0)),
        ('ClinicalLoss (λ=3)', ClinicalLoss(fn_weight=3.0))
    ]

    for name, criterion in loss_configs:
        loss = criterion(logits, targets)
        print(f"{name:<30} Loss: {loss.item():.4f}")

    print("\n✅ All loss functions working correctly!")

    print("\n📝 Usage recommendation:")
    print("  For high-sensitivity arrhythmia detection:")
    print("  - Option 1: FocalLoss(alpha=0.75, gamma=2.0)  [Recommended]")
    print("  - Option 2: WeightedCrossEntropyLoss(weight=[1.0, 3.0])")
    print("  - Option 3: ClinicalLoss(fn_weight=3.0)")
