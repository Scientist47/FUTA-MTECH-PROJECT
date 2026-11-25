"""External validation on PhysioNet/CinC 2021 without retraining
(thesis Section 3.5.2).
"""
from training.train import predict

from .classification_metrics import metrics_with_ci


def testable_classes(labels, class_names, min_positive=100):
    """Superclasses with at least `min_positive` positive records."""
    return [c for c, n in enumerate(class_names) if labels[:, c].sum() >= min_positive]


def evaluate_external(model, X, y, thresholds, class_names, min_positive=100, n_boot=1000):
    """Return the Section 3.5.1 metrics on the external data (testable classes only).

    X must already be filtered and z-scored with the PTB-XL training statistics.
    """
    keep = testable_classes(y, class_names, min_positive)
    print("Testable superclasses:", [class_names[c] for c in keep])
    p = predict(model, X)
    return metrics_with_ci(y[:, keep], p[:, keep], [thresholds[c] for c in keep],
                           [class_names[c] for c in keep], n_boot=n_boot)
