"""Grad-CAM from the last convolutional layer (thesis Section 3.3.5).
Non-negative time map, up-sampled to the signal length, no per-lead values.
"""


def explain_gradcam(model, inputs, target_class, layer):
    """Return Grad-CAM maps of shape (n, 1000)."""
    raise NotImplementedError
