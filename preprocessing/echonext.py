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


NEEDED_FILES = ["EchoNext_metadata_100k.csv", "EchoNext_train_waveforms.npy",
                "EchoNext_val_waveforms.npy", "EchoNext_test_waveforms.npy"]


def prepare_echonext(root, out_dir, prep_fn, chunk=2000):
    """Filter/resample every split in chunks, z-score with training statistics,
    and save float16 arrays X_<split>.npy, y_<split>.npy and meta_<split>.csv.

    Works chunk by chunk from memory-mapped files, so the 250 Hz float32
    waveforms (several GB) never have to fit in RAM at once.
    prep_fn: maps (n, 12, 2500) at 250 Hz -> (n, 12, 1000) at 100 Hz.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_all = pd.read_csv(os.path.join(root, "EchoNext_metadata_100k.csv"))
    filtered, s1, s2, n = {}, None, None, 0
    for split in ["train", "val", "test"]:
        meta = meta_all[meta_all["split"] == split].reset_index(drop=True)
        wav = np.load(os.path.join(root, f"EchoNext_{split}_waveforms.npy"), mmap_mode="r")
        if len(wav) != len(meta):
            raise ValueError(f"{split}: {len(wav)} waveforms but {len(meta)} metadata rows")
        out = np.empty((len(wav), 12, 1000), dtype=np.float16)
        for i in range(0, len(wav), chunk):
            x = np.asarray(wav[i:i + chunk, 0], dtype=np.float32).transpose(0, 2, 1)
            xf = prep_fn(x)
            out[i:i + len(xf)] = xf
            if split == "train":                       # running per-lead sums (training only)
                a = xf.astype(np.float64)
                s1 = a.sum(axis=(0, 2)) if s1 is None else s1 + a.sum(axis=(0, 2))
                s2 = (a ** 2).sum(axis=(0, 2)) if s2 is None else s2 + (a ** 2).sum(axis=(0, 2))
                n += a.shape[0] * a.shape[2]
        filtered[split] = (out, meta)
        y = meta[LABEL].to_numpy(dtype=np.float32)
        print(f"{split}: {out.shape}, SHD positives {int(y.sum()):,} ({100 * y.mean():.1f}%)")
    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
    np.savez(os.path.join(out_dir, "lead_stats.npz"), mean=mean, std=std)
    for split, (out, meta) in filtered.items():
        for i in range(0, len(out), chunk):
            out[i:i + chunk] = ((out[i:i + chunk].astype(np.float32) - mean[None, :, None])
                                / std[None, :, None]).astype(np.float16)
        np.save(os.path.join(out_dir, f"X_{split}.npy"), out)
        np.save(os.path.join(out_dir, f"y_{split}.npy"), meta[LABEL].to_numpy(dtype=np.float32)[:, None])
        meta.to_csv(os.path.join(out_dir, f"meta_{split}.csv"), index=False)
    return {"mean": mean, "std": std}
