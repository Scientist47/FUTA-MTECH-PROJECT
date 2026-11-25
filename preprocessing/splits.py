"""Patient-level partitioning (thesis Section 3.4.2).

Folds 1-8: training. Fold 9: validation, early stopping and thresholds.
Fold 10: test, used once after the hypotheses are fixed.
"""


def split_by_fold(metadata):
    """Return train, validation and test indices from strat_fold."""
    raise NotImplementedError


def check_no_patient_overlap(metadata, splits):
    """Raise an error if any patient_id appears in more than one split."""
    raise NotImplementedError
