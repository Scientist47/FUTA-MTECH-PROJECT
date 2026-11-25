"""SHAP GradientExplainer for the deep architectures (thesis Section 3.3.5).
Signed attribution per sample and per lead, with a background sample of
training records. KernelSHAP is used only for the EBM's tabular features.
"""


def explain_shap(model, inputs, background, target_class):
    """Return attributions of shape (n, 12, 1000)."""
    raise NotImplementedError
