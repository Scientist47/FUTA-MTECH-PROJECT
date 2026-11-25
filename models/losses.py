"""Class-weighted binary cross-entropy (thesis Equation 3.1).

The positive-class weight w_c is set from each class's frequency in the
training folds. For EchoNext, C = 1.
"""


def class_weights(train_labels):
    """Return the positive-class weight for each class."""
    raise NotImplementedError


def weighted_bce(logits, targets, weights):
    """Return the class-weighted binary cross-entropy loss."""
    raise NotImplementedError
