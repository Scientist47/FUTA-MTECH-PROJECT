"""Segment-by-lead alignment (thesis Equation 2).

A_{s,l} = sum over t in segment s of |phi_l(t)|, for five segments
(P, PR, QRS, ST, T) x 12 leads, normalised so the 60 cells sum to one.

Segment boundaries come from PTB-XL+ fiducial points where available
(`mask_from_wfdb_annotation`), with NeuroKit2 as the fallback
(`mask_from_neurokit`). Each record's boundaries are stored as a
"segment mask": an integer array, one value per sample, 0=P, 1=PR, 2=QRS,
3=ST, 4=T and -1 for samples outside any segment (e.g. the T-P baseline).
Maps without lead information (Grad-CAM, attention) are applied to every lead.
"""
import warnings

import numpy as np

SEGMENTS = ["P", "PR", "QRS", "ST", "T"]

# Template offsets from the R peak in seconds (used only if delineation of a
# wave fails): typical adult intervals.
_TEMPLATE = {"P_on": -0.20, "P_off": -0.12, "QRS_on": -0.05, "QRS_off": 0.05,
             "T_on": 0.15, "T_off": 0.40}


def _fill(mask, beats):
    """Write the five segments of each beat into the mask."""
    n = len(mask)
    order = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
    for b in beats:
        pts = [b.get(k) for k in order]
        if any(p is None or np.isnan(p) for p in pts):
            continue
        pts = [int(np.clip(round(p), 0, n)) for p in pts]
        if not all(pts[i] <= pts[i + 1] for i in range(5)):
            continue
        for s in range(5):
            mask[pts[s]:pts[s + 1]] = s
    return mask


def mask_from_rpeaks(rpeaks, n_samples, fs):
    """Template segments placed around each R peak."""
    beats = [{k: r + v * fs for k, v in _TEMPLATE.items()} for r in rpeaks]
    return _fill(np.full(n_samples, -1, dtype=np.int8), beats)


def mask_from_neurokit(lead_signal, fs=100):
    """Delineate one lead (lead II recommended) with NeuroKit2.

    Wave boundaries NeuroKit2 cannot find are filled from the R-peak template,
    so every detected beat gets five segments.
    """
    import neurokit2 as nk
    n = len(lead_signal)
    mask = np.full(n, -1, dtype=np.int8)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(lead_signal, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
            rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
            if len(rpeaks) < 2:
                return mask
            try:
                _, waves = nk.ecg_delineate(clean, rpeaks.astype(int), sampling_rate=fs,
                                            method="dwt")
            except Exception:
                waves = {}
    except Exception:
        return mask
    keys = {"P_on": "ECG_P_Onsets", "P_off": "ECG_P_Offsets", "QRS_on": "ECG_R_Onsets",
            "QRS_off": "ECG_R_Offsets", "T_on": "ECG_T_Onsets", "T_off": "ECG_T_Offsets"}
    beats = []
    for i, r in enumerate(rpeaks):
        b = {}
        for k, nk_key in keys.items():
            vals = waves.get(nk_key)
            v = float(vals[i]) if vals is not None and i < len(vals) else np.nan
            b[k] = v if not np.isnan(v) else r + _TEMPLATE[k] * fs
        beats.append(b)
    return _fill(mask, beats)


def mask_from_wfdb_annotation(ann, n_samples, fs):
    """Build a mask from a WFDB delineation annotation (e.g. PTB-XL+ ECGDeli).

    Assumes the usual WFDB convention: '(' wave onset, then the wave symbol
    ('p', 'N' for QRS, 't'), then ')' wave offset. Sample positions are
    rescaled from the annotation's sampling rate to fs. Check the output on a
    few records (notebook 04) before relying on it.
    """
    scale = fs / float(ann.fs) if getattr(ann, "fs", None) else 1.0
    samples = np.asarray(ann.sample) * scale
    sym = list(ann.symbol)
    waves = []   # (wave, onset, offset)
    for i, s in enumerate(sym):
        if s in ("p", "N", "t"):
            on = samples[i - 1] if i > 0 and sym[i - 1] == "(" else np.nan
            off = samples[i + 1] if i + 1 < len(sym) and sym[i + 1] == ")" else np.nan
            waves.append((s, on, off))
    beats, cur = [], {}
    for s, on, off in waves:
        if s == "p":
            cur = {"P_on": on, "P_off": off}
        elif s == "N":
            cur.update({"QRS_on": on, "QRS_off": off})
        elif s == "t":
            cur.update({"T_on": on, "T_off": off})
            beats.append(cur)
            cur = {}
    return _fill(np.full(n_samples, -1, dtype=np.int8), beats)


def segment_boundaries(signal, fs=100, lead=1, annotation=None):
    """Return the segment mask (n_samples,) for one record.

    signal: (12, n_samples). Uses the annotation (PTB-XL+) when given and it
    yields segments; otherwise NeuroKit2 on `lead` (index 1 = lead II).
    """
    n = signal.shape[-1]
    if annotation is not None:
        m = mask_from_wfdb_annotation(annotation, n, fs)
        if (m >= 0).any():
            return m
    # Lead II first; if delineation finds too little (e.g. an inverted or noisy
    # lead II), try the inverted lead II, then V5 and lead I, and keep the best.
    best = np.full(n, -1, dtype=np.int8)
    for x in (signal[lead], -signal[lead], signal[10], signal[0]):
        m = mask_from_neurokit(x, fs=fs)
        if (m >= 0).mean() > (best >= 0).mean():
            best = m
        if (best >= 0).mean() >= 0.5:
            break
    return best


def batch_boundaries(signals, fs=100, lead=1, verbose=True):
    """Segment masks for many records: (n, n_samples) int8."""
    out = np.full((len(signals), signals.shape[-1]), -1, dtype=np.int8)
    for i, s in enumerate(signals):
        out[i] = segment_boundaries(s, fs=fs, lead=lead)
        if verbose and (i + 1) % 500 == 0:
            print(f"  delineated {i + 1:,} / {len(signals):,}")
    return out


def align_to_grid(attribution, mask, n_leads=12):
    """Equation 2: return a normalised 5 x 12 attribution matrix.

    attribution: (12, n_samples) signed map (SHAP) or (n_samples,) map without
    lead information (Grad-CAM, attention), which is copied to every lead.
    mask: (n_samples,) segment mask from segment_boundaries().
    """
    phi = np.abs(np.asarray(attribution, dtype=np.float64))
    if phi.ndim == 1:
        phi = np.tile(phi, (n_leads, 1))
    A = np.zeros((len(SEGMENTS), phi.shape[0]))
    for s in range(len(SEGMENTS)):
        sel = mask == s
        if sel.any():
            A[s] = phi[:, sel].sum(axis=1)
    total = A.sum()
    return (A / total if total > 0 else A).astype(np.float32)


def align_batch(attributions, masks):
    """align_to_grid for many records: returns (n, 5, 12)."""
    return np.stack([align_to_grid(a, m) for a, m in zip(attributions, masks)])
