"""Class-weighted binary cross-entropy (thesis Equation 1).

L = -1/N sum_n sum_c [ w_c y_nc log yhat_nc + (1 - y_nc) log(1 - yhat_nc) ]

The positive-class weight w_c is set from each class's frequency in the
training folds: w_c = (number of negatives) / (number of positives). For
EchoNext, C = 1.

The loss is computed from logits (the values before the sigmoid) for
numerical stability; this is mathematically the same as Equation 1.
"""
import numpy as np
import torch
import torch.nn.functional as F


def class_weights(train_labels):
    """Return w_c = negatives / positives for each class, as a float tensor."""
    y = np.asarray(train_labels, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = neg / np.maximum(pos, 1.0)
    return torch.tensor(w, dtype=torch.float32)


def weighted_bce(logits, targets, weights):
    """Equation 1: class-weighted binary cross-entropy, averaged over the batch.

    Sums over classes and averages over records (N), as written in Eq. 1.
    """
    per_element = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weights.to(logits.device), reduction="none")
    return per_element.sum(dim=1).mean()


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Focal loss, used only for the preliminary comparison in Chapter 4."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).sum(dim=1).mean()
