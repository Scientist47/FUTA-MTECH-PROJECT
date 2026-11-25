"""Train one architecture on one dataset with one seed (thesis Section 3.4.3).

Adam optimiser, reduce-on-plateau learning rate, batch size 64-128, early
stopping on validation macro-AUROC, best checkpoint kept. Augmentation
(time shift, amplitude scaling, added noise) on training batches only.
Run with five seeds per architecture.

Usage (planned):
    python -m training.train --config training/configs/ptbxl_cnn1d.yaml --seed 0
"""


def set_seed(seed):
    """Fix Python, NumPy and PyTorch random seeds."""
    raise NotImplementedError


def augment(batch):
    """Apply time shift, amplitude scaling and noise to a training batch."""
    raise NotImplementedError


def train(config, seed):
    """Train a model and save the best checkpoint."""
    raise NotImplementedError


def choose_thresholds(model, val_loader):
    """Choose per-class thresholds on fold 9 to maximise F1."""
    raise NotImplementedError
