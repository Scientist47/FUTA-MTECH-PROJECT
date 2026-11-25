"""Interpretable comparator: Explainable Boosting Machine on PTB-XL+ features
(thesis Section 3.3.6). Trained on the same folds; explained by its shape
functions and by KernelSHAP. A reference point only, not part of the model.
"""
import os

import numpy as np
import pandas as pd


def load_ptbxl_plus_features(root, ecg_ids, table="12sl_features.csv", max_missing=0.3):
    """Return a DataFrame of numeric features, one row per ecg_id (in order).

    root: the PTB-XL+ folder (containing features/). Columns with more than
    max_missing missing values are dropped; remaining gaps are filled with the
    training median later (see impute()).
    """
    df = pd.read_csv(os.path.join(root, "features", table))
    id_col = next((c for c in df.columns if "ecg_id" in c.lower()), df.columns[0])
    df = df.set_index(id_col)
    df = df.select_dtypes(include=[np.number])
    df = df.reindex(ecg_ids)
    keep = df.columns[df.isna().mean() <= max_missing]
    return df[keep]


def impute(train_df, *others):
    """Fill missing values with the training-fold median (no leakage)."""
    med = train_df.median()
    return [train_df.fillna(med)] + [o.fillna(med) for o in others]


def train_ebm(features, labels, class_names, seed=0, **kwargs):
    """Fit one EBM per superclass with InterpretML. Returns a dict name -> model."""
    from interpret.glassbox import ExplainableBoostingClassifier
    models = {}
    for c, name in enumerate(class_names):
        ebm = ExplainableBoostingClassifier(random_state=seed, **kwargs)
        ebm.fit(features, labels[:, c])
        models[name] = ebm
        print(f"  EBM trained for {name}")
    return models


def predict_ebm(models, features):
    """Return probabilities (n, n_classes) in the order of the dict."""
    return np.stack([m.predict_proba(features)[:, 1] for m in models.values()], axis=1)


def kernel_shap_ebm(model, background, X, nsamples=200):
    """KernelSHAP attributions for one EBM (tabular features)."""
    import shap
    f = lambda d: model.predict_proba(pd.DataFrame(d, columns=background.columns))[:, 1]
    bg = shap.kmeans(background, 25)
    explainer = shap.KernelExplainer(f, bg)
    return explainer.shap_values(X, nsamples=nsamples)
