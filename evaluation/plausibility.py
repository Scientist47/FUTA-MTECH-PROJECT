"""Clinical plausibility analysis (thesis Section 3.5.4, RQ3 and H3).

Delta = mean rating of real maps - mean rating of control maps (Equation 4),
with a 95% confidence interval; Jaccard overlap of rater-marked cells with
the model's top-k cells; inter-rater agreement (weighted Cohen's kappa or
Krippendorff's alpha).

Ratings are kept in a "long" table with one row per rater x map:
    case_id, map_id, rater, map_type ('real' or 'control'), rating (1-5),
    useful_in_practice ('Yes'/'No')
Rater data are collected only after HREC approval and are never committed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def make_rating_packet(case_ids, seed=0):
    """Blind the study: each case shown twice (real + control) in random order.

    Returns (sheet, key). `sheet` (map_id, case_id) goes to the raters;
    `key` (map_id, case_id, map_type) stays with you and is opened only at
    analysis time.
    """
    rng = np.random.default_rng(seed)
    rows = [(c, t) for c in case_ids for t in ("real", "control")]
    order = rng.permutation(len(rows))
    key = pd.DataFrame([rows[i] for i in order], columns=["case_id", "map_type"])
    key.insert(0, "map_id", [f"M{i + 1:03d}" for i in range(len(key))])
    sheet = key[["map_id", "case_id"]].copy()
    sheet["rating_1_to_5"] = ""
    sheet["useful_in_nigerian_practice_yes_no"] = ""
    sheet["comments"] = ""
    return sheet, key


def rating_difference(ratings, n_boot=5000, seed=0):
    """Equation 4: Delta = mean(real) - mean(control), with a 95% CI.

    The CI resamples cases (each case keeps its real and control maps and all
    raters), so repeated ratings of the same case are not treated as independent.
    """
    df = ratings.copy()
    per_case = df.pivot_table(index="case_id", columns="map_type", values="rating", aggfunc="mean")
    d = (per_case["real"] - per_case["control"]).to_numpy()
    delta = float(df.loc[df.map_type == "real", "rating"].mean()
                  - df.loc[df.map_type == "control", "rating"].mean())
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from scipy.stats import wilcoxon
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"delta": delta, "ci_low": float(lo), "ci_high": float(hi),
            "n_cases": int(len(d)), "wilcoxon_p": p}


def jaccard_overlap(rater_cells, model_grid, k=None):
    """Jaccard index between the rater's marked cells and the model's top-k cells.

    rater_cells: 5 x 12 array of 0/1 marks. k defaults to the number of cells
    the rater marked.
    """
    r = np.asarray(rater_cells).reshape(-1).astype(bool)
    k = int(k or max(r.sum(), 1))
    m = np.zeros_like(r)
    m[np.argsort(-np.asarray(model_grid).reshape(-1))[:k]] = True
    union = (r | m).sum()
    return float((r & m).sum() / union) if union else 0.0


def krippendorff_alpha_interval(matrix):
    """Krippendorff's alpha (interval metric) for a raters x items matrix.

    NaN marks a missing rating. The interval metric uses squared
    differences, the counterpart of quadratic weighting in kappa.
    """
    data = np.asarray(matrix, dtype=float)
    values, pairs_d, n_pairable = [], 0.0, 0
    units = [col[~np.isnan(col)] for col in data.T]
    units = [u for u in units if len(u) >= 2]
    all_vals = np.concatenate(units)
    n = len(all_vals)
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    de = ((all_vals[:, None] - all_vals[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else 1.0


def inter_rater_agreement(ratings):
    """Quadratic-weighted Cohen's kappa (2 raters) or Krippendorff's alpha (3+)."""
    wide = ratings.pivot_table(index="map_id", columns="rater", values="rating")
    raters = list(wide.columns)
    if len(raters) == 2:
        both = wide.dropna()
        k = cohen_kappa_score(both[raters[0]].astype(int), both[raters[1]].astype(int),
                              weights="quadratic")
        return {"measure": "quadratic-weighted Cohen's kappa", "value": float(k)}
    return {"measure": "Krippendorff's alpha (interval)",
            "value": krippendorff_alpha_interval(wide.T.to_numpy())}
