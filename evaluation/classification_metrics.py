"""Classification performance (thesis Section 3.5.1, RQ1 and H1).

Macro-AUROC (primary), per-class AUROC, F1, sensitivity and specificity,
expected calibration error and Brier score, all with 95% bootstrap
confidence intervals. Paired bootstrap and DeLong tests between models.
"""


def compute_metrics(y_true, y_prob, thresholds):
    """Return the metric table for one model."""
    raise NotImplementedError


def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=1000):
    """Return a 95% bootstrap confidence interval."""
    raise NotImplementedError


def delong_test(y_true, prob_a, prob_b):
    """Compare two correlated AUROCs."""
    raise NotImplementedError
