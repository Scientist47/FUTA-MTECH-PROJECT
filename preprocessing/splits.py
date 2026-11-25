"""Patient-level partitioning (thesis Section 3.4.2).

Folds 1-8: training. Fold 9: validation, early stopping and thresholds.
Fold 10: test, used once after the hypotheses are fixed.
"""
import numpy as np


def split_by_fold(metadata, train_folds=range(1, 9), val_fold=9, test_fold=10):
    """Return a dict of integer row positions for 'train', 'val' and 'test'."""
    fold = metadata["strat_fold"].to_numpy()
    return {
        "train": np.where(np.isin(fold, list(train_folds)))[0],
        "val": np.where(fold == val_fold)[0],
        "test": np.where(fold == test_fold)[0],
    }


def check_no_patient_overlap(metadata, splits):
    """Raise an error if any patient_id appears in more than one split."""
    pid = metadata["patient_id"].to_numpy()
    sets = {k: set(pid[v]) for k, v in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = sets[names[i]] & sets[names[j]]
            if shared:
                raise ValueError(
                    f"{len(shared)} patients appear in both {names[i]} and "
                    f"{names[j]}, e.g. {sorted(shared)[:5]}")
    return True
