"""Train one architecture on one dataset with one seed (thesis Section 3.4.3).

Adam optimiser, reduce-on-plateau learning rate, batch size 64-128, early
stopping on validation macro-AUROC, best checkpoint kept. Augmentation
(time shift, amplitude scaling, added noise) on training batches only.
Run with five seeds per architecture.

Usage from a notebook:
    from training.train import train
    result = train(config, seed=0, data=data, out_dir='results/ptbxl_cnn1d')

`data` is a dict with X_train, y_train, X_val, y_val (NumPy arrays, already
filtered and z-scored), as saved by notebook 02.
"""
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from models import build_model
from models.losses import class_weights, focal_loss, weighted_bce


def set_seed(seed):
    """Fix Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment(batch, max_shift=50, scale_range=(0.9, 1.1), noise_std=0.05):
    """Apply time shift, amplitude scaling and Gaussian noise to a batch.

    batch: tensor (B, leads, T). Shift is circular, up to max_shift samples
    (0.5 s at 100 Hz); scaling is one factor per record; noise is in z-score
    units. No interpolation between records (no SMOTE / mixup).
    """
    b = batch.shape[0]
    shifts = torch.randint(-max_shift, max_shift + 1, (b,))
    out = torch.stack([torch.roll(batch[i], int(shifts[i]), dims=-1) for i in range(b)])
    scale = torch.empty(b, 1, 1, device=batch.device).uniform_(*scale_range)
    return out * scale + noise_std * torch.randn_like(out)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, X, batch_size=256, device=None):
    """Return sigmoid probabilities (n, n_outputs) for a NumPy array X."""
    device = device or get_device()
    model.eval().to(device)
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]), dtype=torch.float32, device=device)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def macro_auroc(y_true, y_prob):
    aucs = [roc_auc_score(y_true[:, c], y_prob[:, c])
            for c in range(y_true.shape[1]) if 0 < y_true[:, c].sum() < len(y_true)]
    return float(np.mean(aucs)) if aucs else float("nan")


def choose_thresholds(y_val, p_val):
    """Choose one threshold per class on fold 9 to maximise F1."""
    grid = np.linspace(0.05, 0.95, 91)
    th = []
    for c in range(y_val.shape[1]):
        scores = [f1_score(y_val[:, c], p_val[:, c] >= t, zero_division=0) for t in grid]
        th.append(float(grid[int(np.argmax(scores))]))
    return np.array(th)


def train(config, seed, data, out_dir, verbose=True):
    """Train a model, keep the best checkpoint and return a summary dict.

    config keys: architecture, n_outputs, batch_size, lr, max_epochs,
    patience, loss ('weighted_bce' or 'focal'), augment (bool).
    Saves <out_dir>/seed<seed>/best.pt, history.json and thresholds.json.
    """
    set_seed(seed)
    device = get_device()
    run_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    if ytr.ndim == 1:
        ytr, yva = ytr[:, None], yva[:, None]
    model = build_model(config["architecture"], Xtr.shape[1], config["n_outputs"]).to(device)
    weights = class_weights(ytr).to(device)
    loss_name = config.get("loss", "weighted_bce")

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr)),
                        batch_size=config.get("batch_size", 64), shuffle=True,
                        generator=g, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_auc, best_epoch, wait, history = -1.0, 0, 0, []
    for epoch in range(1, config.get("max_epochs", 50) + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            if config.get("augment", True):
                xb = augment(xb)
            logits = model(xb)
            loss = weighted_bce(logits, yb, weights) if loss_name == "weighted_bce" \
                else focal_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(loader.dataset)

        p_val = predict(model, Xva, device=device)
        with torch.no_grad():
            val_loss = weighted_bce(torch.logit(torch.as_tensor(p_val).clamp(1e-6, 1 - 1e-6)),
                                    torch.as_tensor(yva), weights.cpu()).item()
        val_auc = macro_auroc(yva, p_val)
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_auroc": val_auc, "lr": opt.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"val_macro_AUROC {val_auc:.4f}  ({history[-1]['seconds']}s)")
        if val_auc > best_auc:
            best_auc, best_epoch, wait = val_auc, epoch, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            wait += 1
            if wait >= config.get("patience", 8):
                if verbose:
                    print(f"early stop: no improvement for {wait} epochs")
                break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    thresholds = choose_thresholds(yva, predict(model, Xva, device=device))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(run_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds.tolist(), f)
    summary = {"seed": seed, "best_epoch": best_epoch, "best_val_macro_auroc": best_auc,
               "thresholds": thresholds.tolist(), "run_dir": run_dir}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return model, summary


def load_trained(architecture, n_outputs, run_dir, n_leads=12, device=None):
    """Load a saved best checkpoint and its thresholds."""
    device = device or get_device()
    model = build_model(architecture, n_leads, n_outputs)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    with open(os.path.join(run_dir, "thresholds.json")) as f:
        th = np.array(json.load(f))
    return model.to(device).eval(), th
