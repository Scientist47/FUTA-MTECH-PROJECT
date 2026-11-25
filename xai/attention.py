"""Attention weights of the CNN-BiLSTM attention-pooling layer
(thesis Section 3.3.5). One weight per time step, shared across leads.
"""
import numpy as np
import torch

from ._utils import to_tensor, upsample_time


@torch.no_grad()
def extract_attention(model, inputs, batch_size=128):
    """Return attention weights up-sampled to shape (n, n_samples).

    The 125 attention steps (after three pooling layers) are stretched to
    the 1,000 input samples; each row still sums to (about) one per step.
    """
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(inputs), batch_size):
        _, a = model(to_tensor(inputs[i:i + batch_size], device), return_attention=True)
        out.append(a.cpu().numpy())
    return upsample_time(np.concatenate(out), inputs.shape[-1]).astype(np.float32)
