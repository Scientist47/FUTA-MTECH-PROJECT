"""Signal filtering and normalisation (thesis Section 3.4.1).

- 0.5 Hz high-pass filter (baseline wander).
- 50 Hz notch filter (power-line interference).
- Per-lead z-score with statistics fitted on training folds 1-8 only.

All functions work on arrays shaped (n_records, n_leads, n_samples) or
(n_leads, n_samples); filtering is always along the last axis (time).

Note on the notch filter at 100 Hz: a signal sampled at 100 Hz can only hold
frequencies below 50 Hz (its Nyquist frequency), so 50 Hz interference has
already been removed when PTB-XL was downsampled from 500 Hz to 100 Hz. A
50 Hz notch cannot be designed at fs = 100 Hz, so `notch` applies the filter
only when fs > 2 x freq (for example the 500 Hz ablation, or EchoNext before
resampling) and otherwise returns the signal unchanged with a message.
"""
import numpy as np
from scipy import signal as sps


def highpass(x, fs=100, cutoff=0.5, order=2):
    """Remove baseline wander with a zero-phase Butterworth high-pass filter."""
    sos = sps.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def notch(x, fs=100, freq=50, quality=30.0):
    """Remove power-line interference with a zero-phase notch filter.

    Returns the signal unchanged when freq is at or above the Nyquist
    frequency (fs / 2), because the interference cannot be present there.
    """
    if freq >= fs / 2:
        print(
            f"Note: {freq} Hz notch skipped: at fs = {fs} Hz the highest frequency "
            f"in the signal is {fs / 2} Hz, so {freq} Hz interference was "
            "already removed by the dataset's downsampling.")
        return x.astype(np.float32)
    b, a = sps.iirnotch(freq, quality, fs=fs)
    return sps.filtfilt(b, a, x, axis=-1).astype(np.float32)


def resample(x, fs_in, fs_out):
    """Resample along time (used for EchoNext and the CinC 2021 sources)."""
    if fs_in == fs_out:
        return x.astype(np.float32)
    n_out = int(round(x.shape[-1] * fs_out / fs_in))
    return sps.resample(x, n_out, axis=-1).astype(np.float32)


def fit_lead_stats(train_signals):
    """Return per-lead mean and standard deviation from the training folds.

    train_signals: (n_records, n_leads, n_samples). Returns a dict with
    'mean' and 'std', each shaped (n_leads,).
    """
    mean = train_signals.mean(axis=(0, 2))
    std = train_signals.std(axis=(0, 2))
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_zscore(signals, stats):
    """Standardise each lead with training-fold statistics."""
    m = stats["mean"][None, :, None]
    s = stats["std"][None, :, None]
    return ((signals - m) / s).astype(np.float32)


def clean(x, fs=100):
    """High-pass then notch: the full filtering step of Section 3.4.1."""
    return notch(highpass(x, fs=fs), fs=fs)
