"""SHAP GradientExplainer for the deep architectures (thesis Section 3.3.5).

Signed attribution per sample and per lead, with a background sample of
training records. KernelSHAP is used only for the EBM's tabular features
(see models/ebm_comparator.py). Attributions are for the class logit.
"""
import numpy as np
import shap
import torch
import torch.nn as nn

from ._utils import gradients_through_lstm, to_tensor


class _SingleOutput(nn.Module):
    """Expose one class of a multi-output model as a (batch, 1) output."""

    def __init__(self, model, target_class):
        super().__init__()
        self.model, self.c = model, target_class

    def forward(self, x):
        return self.model(x)[:, self.c:self.c + 1]


def explain_shap(model, inputs, background, target_class, nsamples=200, batch_size=16):
    """Return SHAP attributions of shape (n, 12, 1000) for one class.

    background: about 100 training records (n_bg, 12, 1000).
    nsamples: SHAP samples per record (more = smoother, slower).
    """
    device = next(model.parameters()).device
    model.eval()
    wrapped = _SingleOutput(model, target_class).eval()
    bg = to_tensor(background, device)
    out = []
    with gradients_through_lstm():
        explainer = shap.GradientExplainer(wrapped, bg, batch_size=batch_size)
        for i in range(0, len(inputs), batch_size):
            xb = to_tensor(inputs[i:i + batch_size], device)
            sv = explainer.shap_values(xb, nsamples=nsamples, rseed=0)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            if sv.ndim == 4:            # (n, 12, 1000, 1) in recent shap versions
                sv = sv[..., 0]
            out.append(sv.astype(np.float32))
    return np.concatenate(out)
