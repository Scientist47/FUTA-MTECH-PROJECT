"""Grad-CAM from the last convolutional layer (thesis Section 3.3.5).
Non-negative time map, up-sampled to the signal length, no per-lead values.
"""
import numpy as np
import torch

from ._utils import gradients_through_lstm, to_tensor, upsample_time


def explain_gradcam(model, inputs, target_class, layer=None, batch_size=64):
    """Return Grad-CAM maps of shape (n, n_samples), each scaled to max 1.

    layer defaults to model.gradcam_layer (the last convolutional stage).
    """
    layer = layer or model.gradcam_layer
    device = next(model.parameters()).device
    model.eval()
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    maps = []
    try:
        with gradients_through_lstm():
            for i in range(0, len(inputs), batch_size):
                xb = to_tensor(inputs[i:i + batch_size], device).requires_grad_(True)
                model.zero_grad()
                model(xb)[:, target_class].sum().backward()
                act, grad = store["act"].detach(), store["grad"].detach()   # (B, C, T')
                w = grad.mean(dim=2, keepdim=True)
                cam = torch.relu((w * act).sum(dim=1)).cpu().numpy()       # (B, T')
                maps.append(cam)
    finally:
        h1.remove()
        h2.remove()
    cam = upsample_time(np.concatenate(maps), inputs.shape[-1])
    cam = np.maximum(cam, 0)
    return (cam / np.maximum(cam.max(axis=1, keepdims=True), 1e-12)).astype(np.float32)
