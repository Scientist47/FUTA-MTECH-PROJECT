"""Load PTB-XL records and labels (thesis Sections 3.2.1 and 3.4.1).

- Read the 100 Hz records (12 x 1,000 samples) with wfdb.
- Map scp_codes to the five diagnostic superclasses (NORM, MI, STTC, CD, HYP)
  using scp_statements.csv supplied with the dataset.
- Keep strat_fold and patient_id for partitioning and leakage checks.

Lead order in PTB-XL: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
"""
import ast
import os

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def map_superclasses(scp_codes, statements):
    """Return the multi-hot superclass labels (length 5) for one record.

    scp_codes: dict like {'IMI': 100.0, 'SR': 0.0}. Every diagnostic
    statement in the record is mapped to its diagnostic_class, following
    the PTB-XL benchmark (Strodthoff et al., 2021).
    """
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in scp_codes:
        if code in statements.index:
            cls = statements.loc[code, "diagnostic_class"]
            if isinstance(cls, str) and cls in SUPERCLASSES:
                y[SUPERCLASSES.index(cls)] = 1.0
    return y


def load_metadata(root, drop_unlabelled=True):
    """Return the PTB-XL metadata table with one 0/1 column per superclass.

    Records with no diagnostic superclass are dropped (as in the benchmark),
    unless drop_unlabelled=False.
    """
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    statements = statements[statements["diagnostic"] == 1]
    labels = np.stack([map_superclasses(c, statements) for c in df["scp_codes"]])
    for i, name in enumerate(SUPERCLASSES):
        df[name] = labels[:, i]
    df["n_superclasses"] = labels.sum(axis=1)
    if drop_unlabelled:
        df = df[df["n_superclasses"] > 0].copy()
    df["strat_fold"] = df["strat_fold"].astype(int)
    return df


def load_signals(root, metadata, sampling_rate=100, verbose=True):
    """Return float32 array (n_records, 12, n_samples) in millivolts.

    Uses filename_lr (100 Hz, 1,000 samples) or filename_hr (500 Hz, 5,000).
    """
    import wfdb
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    files = metadata[col].tolist()
    out = None
    for i, f in enumerate(files):
        sig, _ = wfdb.rdsamp(os.path.join(root, f))
        if out is None:
            out = np.zeros((len(files), sig.shape[1], sig.shape[0]), dtype=np.float32)
        out[i] = sig.T
        if verbose and (i + 1) % 2000 == 0:
            print(f"  read {i + 1:,} / {len(files):,} records")
    return np.nan_to_num(out)


def labels_array(metadata):
    """Return the (n_records, 5) label matrix in SUPERCLASSES order."""
    return metadata[SUPERCLASSES].to_numpy(dtype=np.float32)
