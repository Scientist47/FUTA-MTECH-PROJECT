"""Faithfulness of the explanations (thesis Section 3.5.3, RQ2 and H2).

- Deletion and insertion curves over segment-by-lead cells (Equation 3),
  with removed cells replaced by zero (the training mean after normalisation).
- Randomisation checks (layer re-initialisation, shuffled labels).
- Stability under small perturbations.
- Cross-method agreement (Spearman correlation, top-k overlap).
- Known-answer tests (QRS-duration regression, planted waveform).
All compared against a random-attribution baseline, on 500 test records.

Terminology: a "grid" is the 5 x 12 matrix from xai.segment_alignment
(Equation 2); a "mask" is the per-sample segment mask of one record.
"""
import copy

import numpy as np
import torch
from scipy.stats import spearmanr

from xai._utils import to_tensor


def _cell_masks(mask, n_leads=12):
    """Boolean (60, 12, T) array: which samples belong to each cell."""
    T = len(mask)
    cells = np.zeros((5 * n_leads, n_leads, T), dtype=bool)
    for s in range(5):
        seg = mask == s
        for l in range(n_leads):
            cells[s * n_leads + l, l, seg] = True
    return cells


def _order(grid, rng=None):
    """Cell indices (s*12 + l), most important first; random if rng given."""
    flat = np.asarray(grid).reshape(-1)
    if rng is not None:
        return rng.permutation(len(flat))
    return np.argsort(-flat, kind="stable")


@torch.no_grad()
def _curve(model, x, mask, order, target_class, mode, step=1):
    cells = _cell_masks(mask, x.shape[0])
    K = int(np.ceil(len(order) / step))
    batch = np.empty((K + 1,) + x.shape, dtype=np.float32)
    cur = x.copy() if mode == "deletion" else np.zeros_like(x)
    batch[0] = cur
    for k in range(1, K + 1):
        idx = order[(k - 1) * step:k * step]
        sel = cells[idx].any(axis=0)
        cur = cur.copy()
        cur[sel] = 0.0 if mode == "deletion" else x[sel]
        batch[k] = cur
    device = next(model.parameters()).device
    model.eval()
    out = torch.sigmoid(model(to_tensor(batch, device)))[:, target_class].cpu().numpy()
    return out


def deletion_curve(model, x, mask, grid, target_class, step=1, rng=None):
    """Model output for class c after removing 0, 1, ..., K cells."""
    return _curve(model, x, mask, _order(grid, rng), target_class, "deletion", step)


def insertion_curve(model, x, mask, grid, target_class, step=1, rng=None):
    """Model output after adding back 0, 1, ..., K cells to a zero record."""
    return _curve(model, x, mask, _order(grid, rng), target_class, "insertion", step)


def area(curve):
    """Equation 3: AUC = (1/K) * sum_{k=0..K} f_c(x^(k)), K = number of steps."""
    K = len(curve) - 1
    return float(np.sum(curve) / K)


def deletion_auc(model, inputs, masks, grids, target_classes, step=1, seed=0):
    """Deletion AUC for each record, for the explanation and for random order.

    Returns a dict of arrays: 'explanation', 'random'. A faithful method has a
    LOWER explanation AUC than random.
    """
    rng = np.random.default_rng(seed)
    ex, rd = [], []
    for x, m, g, c in zip(inputs, masks, grids, target_classes):
        ex.append(area(deletion_curve(model, x, m, g, c, step)))
        rd.append(area(deletion_curve(model, x, m, g, c, step, rng=rng)))
    return {"explanation": np.array(ex), "random": np.array(rd)}


def insertion_auc(model, inputs, masks, grids, target_classes, step=1, seed=0):
    """Insertion AUC; a faithful method has a HIGHER value than random."""
    rng = np.random.default_rng(seed)
    ex, rd = [], []
    for x, m, g, c in zip(inputs, masks, grids, target_classes):
        ex.append(area(insertion_curve(model, x, m, g, c, step)))
        rd.append(area(insertion_curve(model, x, m, g, c, step, rng=rng)))
    return {"explanation": np.array(ex), "random": np.array(rd)}


def random_grids(n, seed=0):
    """Random-attribution baseline: random non-negative grids summing to one."""
    g = np.random.default_rng(seed).random((n, 5, 12))
    return g / g.sum(axis=(1, 2), keepdims=True)


def grid_similarity(grids_a, grids_b):
    """Spearman correlation between paired grids (one value per record)."""
    out = []
    for a, b in zip(grids_a, grids_b):
        r = spearmanr(np.ravel(a), np.ravel(b)).correlation
        out.append(0.0 if np.isnan(r) else r)
    return np.array(out)


def topk_overlap(grid_a, grid_b, k=10):
    """Fraction of the top-k cells shared by two grids."""
    ta = set(np.argsort(-np.ravel(grid_a))[:k])
    tb = set(np.argsort(-np.ravel(grid_b))[:k])
    return len(ta & tb) / k


def method_agreement(grids_a, grids_b, k=10):
    """Return mean Spearman correlation and mean top-k overlap."""
    rho = grid_similarity(grids_a, grids_b)
    ov = np.array([topk_overlap(a, b, k) for a, b in zip(grids_a, grids_b)])
    return {"spearman_mean": float(rho.mean()), "topk_overlap_mean": float(ov.mean()),
            "spearman": rho, "topk_overlap": ov}


def _layers_top_down(model):
    """Parameterised leaf layers, from the output back to the input."""
    layers = [m for m in model.modules()
              if len(list(m.children())) == 0 and any(True for _ in m.parameters(recurse=False))]
    return layers[::-1]


def randomisation_check(model, explain_fn, inputs, masks, n_steps=None):
    """Cascading layer re-initialisation (Adebayo et al., 2018).

    explain_fn(model, inputs) must return grids (n, 5, 12) for the inputs,
    e.g. lambda m, X: align_batch(explain_shap(m, X, bg, c), masks).
    Returns a list of (n_layers_randomised, mean Spearman vs original).
    A faithful method shows similarity falling well below 1.
    """
    original = explain_fn(model, inputs)
    rand = copy.deepcopy(model)
    layers = _layers_top_down(rand)
    n_steps = n_steps or len(layers)
    results = []
    torch.manual_seed(123)
    for i, layer in enumerate(layers[:n_steps], start=1):
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        if i in (1, 2, 4, 8, 16, 32) or i == n_steps:
            sim = grid_similarity(original, explain_fn(rand, inputs)).mean()
            results.append((i, float(sim)))
            print(f"  {i:3d} layers randomised: mean Spearman = {sim:.3f}")
    return results


def stability(model, explain_fn, inputs, noise_std=0.02, seed=0):
    """Similarity of grids after adding small noise that leaves predictions unchanged."""
    rng = np.random.default_rng(seed)
    noisy = (inputs + noise_std * rng.standard_normal(inputs.shape)).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        p0 = torch.sigmoid(model(to_tensor(inputs, device))).cpu().numpy() >= 0.5
        p1 = torch.sigmoid(model(to_tensor(noisy, device))).cpu().numpy() >= 0.5
    keep = (p0 == p1).all(axis=1)
    sims = grid_similarity(explain_fn(model, inputs[keep]), explain_fn(model, noisy[keep]))
    return {"n_kept": int(keep.sum()), "spearman_mean": float(sims.mean()), "spearman": sims}


def segment_share(grids, segment):
    """Share of total attribution falling in one segment (0=P ... 4=T)."""
    g = np.asarray(grids)
    return g[:, segment, :].sum(axis=1)


def plant_waveform(signals, masks, segment=3, amplitude=1.0, seed=0):
    """Known-answer data: add a smooth bump inside one segment of half the records.

    Returns (new_signals, labels) where label 1 = bump present. The bump is
    placed in the chosen segment (default ST) of every beat, in all leads.
    """
    rng = np.random.default_rng(seed)
    X = signals.copy()
    y = (rng.random(len(X)) < 0.5).astype(np.float32)
    for i in np.where(y == 1)[0]:
        seg = masks[i] == segment
        if not seg.any():
            y[i] = 0
            continue
        # smooth bump over each contiguous run of the segment
        idx = np.where(seg)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            X[i][:, r] += amplitude * np.hanning(len(r) + 2)[1:-1]
    return X, y[:, None]


def fit_quick(model, X, y, loss="bce", epochs=5, lr=1e-3, batch_size=64, seed=0):
    """Small training loop for the known-answer models (not for the main results)."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            j = perm[i:i + batch_size]
            out = model(Xt[j].to(device))
            tgt = yt[j].to(device)
            l = F.binary_cross_entropy_with_logits(out, tgt) if loss == "bce" else F.mse_loss(out, tgt)
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * len(j)
        print(f"  epoch {ep + 1}: loss {tot / len(Xt):.4f}")
    return model.eval()
