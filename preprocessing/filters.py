"""Signal filtering and normalisation (thesis Section 3.4.1).

- 0.5 Hz high-pass filter (baseline wander).
- 50 Hz notch filter (power-line interference).
- Per-lead z-score with statistics fitted on training folds 1-8 only.
"""


def highpass(signal, fs=100, cutoff=0.5):
    """Remove baseline wander."""
    raise NotImplementedError


def notch(signal, fs=100, freq=50):
    """Remove power-line interference."""
    raise NotImplementedError


def fit_lead_stats(train_signals):
    """Return per-lead mean and standard deviation from the training folds."""
    raise NotImplementedError


def apply_zscore(signals, stats):
    """Standardise each lead with training-fold statistics."""
    raise NotImplementedError
