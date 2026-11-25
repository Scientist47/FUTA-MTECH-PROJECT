"""Classification performance (thesis Section 3.5.1, RQ1 and H1).

Macro-AUROC (primary), per-class AUROC, F1, sensitivity and specificity,
expected calibration error and Brier score, all with 95% bootstrap
confidence intervals. Paired bootstrap and DeLong tests between models.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def expected_calibration_error(y, p, n_bins=10):
    """ECE with equal-width bins, for one class."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def _per_class(y, p, t):
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    return {
        "AUROC": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "ECE": expected_calibration_error(y, p),
        "Brier": brier_score_loss(y, p),
    }


def compute_metrics(y_true, y_prob, thresholds, class_names=None):
    """Return the metric table for one model (one row per class + 'Macro')."""
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_prob = np.asarray(y_prob).reshape(len(y_prob), -1)
    class_names = class_names or [f"class{c}" for c in range(y_true.shape[1])]
    rows = {name: _per_class(y_true[:, c], y_prob[:, c], thresholds[c])
            for c, name in enumerate(class_names)}
    df = pd.DataFrame(rows).T
    if len(class_names) > 1:
        df.loc["Macro"] = df.mean()
    return df


def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=1000, seed=0, alpha=0.05):
    """Return (estimate, lower, upper): 95% bootstrap CI over test records."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    est = metric_fn(y_true, y_prob)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            v = metric_fn(y_true[i], y_prob[i])
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


def macro_auroc(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_prob = np.asarray(y_prob).reshape(len(y_prob), -1)
    return float(np.nanmean([roc_auc_score(y_true[:, c], y_prob[:, c])
                             for c in range(y_true.shape[1])
                             if 0 < y_true[:, c].sum() < len(y_true)]))


def metrics_with_ci(y_true, y_prob, thresholds, class_names, n_boot=1000):
    """Table of 'value (lower-upper)' strings, ready for Chapter 4."""
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_prob = np.asarray(y_prob).reshape(len(y_prob), -1)
    rows = {}
    for c, name in enumerate(class_names):
        y, p, t = y_true[:, c], y_prob[:, c], thresholds[c]
        fns = {
            "AUROC": lambda a, b: roc_auc_score(a, b),
            "F1": lambda a, b, t=t: f1_score(a, b >= t, zero_division=0),
            "Sensitivity": lambda a, b, t=t: np.mean(b[a == 1] >= t) if (a == 1).any() else np.nan,
            "Specificity": lambda a, b, t=t: np.mean(b[a == 0] < t) if (a == 0).any() else np.nan,
        }
        rows[name] = {k: "{:.3f} ({:.3f}-{:.3f})".format(*bootstrap_ci(f, y, p, n_boot))
                      for k, f in fns.items()}
    if len(class_names) > 1:
        rows["Macro"] = {"AUROC": "{:.3f} ({:.3f}-{:.3f})".format(
            *bootstrap_ci(macro_auroc, y_true, y_prob, n_boot))}
    return pd.DataFrame(rows).T.fillna("")


def summarise_seeds(values):
    """Mean and 95% t-interval across seeds (e.g. five macro-AUROCs)."""
    v = np.asarray(values, dtype=float)
    m = v.mean()
    if len(v) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(m), float(m - h), float(m + h)


# ---- DeLong test (Sun & Xu, 2014 fast algorithm) ----
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y_true, prob_a, prob_b):
    """Compare two correlated AUROCs on the same records.

    Returns (auc_a, auc_b, z, p_value).
    """
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)
    m = int(y.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    n = preds.shape[1] - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n_boot=2000, seed=0):
    """Mean difference in AUROC (a - b), 95% CI and two-sided p-value."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y_true, prob_a, prob_b))
    diff = roc_auc_score(y, a) - roc_auc_score(y, b)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(diff), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0))
