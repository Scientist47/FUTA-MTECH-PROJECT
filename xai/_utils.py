"""Small helpers shared by the explanation methods."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def gradients_through_lstm():
    """cuDNN cannot back-propagate through an LSTM in eval mode, so turn
    cuDNN off while explanations are computed (the model stays in eval mode,
    so dropout and batch normalisation behave as at test time)."""
    with torch.backends.cudnn.flags(enabled=False):
        yield


def upsample_time(maps, length):
    """Linear interpolation of (n, t) maps to (n, length)."""
    t = torch.as_tensor(maps, dtype=torch.float32).unsqueeze(1)
    return F.interpolate(t, size=length, mode="linear", align_corners=False).squeeze(1).numpy()


def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
