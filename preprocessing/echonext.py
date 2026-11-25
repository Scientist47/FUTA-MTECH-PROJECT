"""Load EchoNext records and the binary structural heart disease (SHD) label
(thesis Sections 3.2.3 and 3.4.4).

EchoNext v1.1.1 on PhysioNet ships NumPy arrays, not WFDB records:
- EchoNext_metadata_100k.csv: one row per ECG, in the same order as the arrays,
  with the label column `shd_moderate_or_greater_flag` and a `split` column
  (train, val, test, no_split).
- EchoNext_<split>_waveforms.npy: shape (N, 1, 2500, 12): 10 s, 12 leads,
  250 Hz, already median-filtered, clipped and normalised by the providers.

Never commit EchoNext data. Check the lead order against the dataset's
documentation; this code assumes the standard order used for PTB-XL.
"""
import os

import numpy as np
import pandas as pd

LABEL = "shd_moderate_or_greater_flag"
FS_ECHONEXT = 250


def load_split(root, split, mmap=True):
    """Return (signals, labels, metadata) for 'train', 'val' or 'test'.

    signals: float32 (N, 12, 2500) at 250 Hz. Use preprocessing.filters.resample
    to bring them to 100 Hz (2500 -> 1000 samples).
    """
    meta = pd.read_csv(os.path.join(root, "EchoNext_metadata_100k.csv"))
    meta = meta[meta["split"] == split].reset_index(drop=True)
    wav = np.load(os.path.join(root, f"EchoNext_{split}_waveforms.npy"),
                  mmap_mode="r" if mmap else None)
    if len(wav) != len(meta):
        raise ValueError(f"{split}: {len(wav)} waveforms but {len(meta)} metadata rows")
    signals = np.asarray(wav[:, 0], dtype=np.float32).transpose(0, 2, 1)  # (N,12,2500)
    labels = meta[LABEL].to_numpy(dtype=np.float32)[:, None]
    return signals, labels, meta
