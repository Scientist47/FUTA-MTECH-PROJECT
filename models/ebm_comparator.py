"""Interpretable comparator: Explainable Boosting Machine on PTB-XL+ features
(thesis Section 3.3.6). Trained on the same folds; explained by its shape
functions and by KernelSHAP. A reference point only, not part of the model.
"""


def train_ebm(features, labels):
    """Fit one EBM per superclass with InterpretML."""
    raise NotImplementedError
