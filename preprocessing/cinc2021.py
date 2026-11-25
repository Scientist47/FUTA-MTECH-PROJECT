"""Load the PhysioNet/CinC 2021 collection for external validation
(thesis Sections 3.2.4 and 3.5.2).

- Exclude the PTB and PTB-XL sources (any source folder whose name contains 'ptb').
- Map SNOMED CT codes to the five superclasses.
- Test only superclasses with at least 100 positive records.

Each record is a WFDB header (.hea) plus a .mat signal file; the diagnoses
are in a header comment line '#Dx: code,code,...'. Records are resampled to
100 Hz and cut (or zero-padded) to 10 s so they match the PTB-XL input.

SNOMED -> superclass map: the PTB-XL+ file labels/mapping/ptbxlToSNOMED.csv
links PTB-XL SCP statements to SNOMED CT codes; combined with
scp_statements.csv it gives each SNOMED code a superclass. Check its column
names the first time (notebook 03 prints them).
"""
import glob
import os

import numpy as np
import pandas as pd

from .filters import resample
from .ptbxl import SUPERCLASSES


def build_snomed_map(ptbxl_plus_root, ptbxl_root):
    """Return {snomed_code (str): superclass} from the PTB-XL+ mapping file."""
    mp = pd.read_csv(os.path.join(ptbxl_plus_root, "labels", "mapping", "ptbxlToSNOMED.csv"))
    print("ptbxlToSNOMED.csv columns:", list(mp.columns))
    scp_col = next(c for c in mp.columns if "snomed" not in c.lower())
    sno_col = next(c for c in mp.columns if "snomed" in c.lower() and "desc" not in c.lower())
    st = pd.read_csv(os.path.join(ptbxl_root, "scp_statements.csv"), index_col=0)
    out = {}
    for scp, sno in zip(mp[scp_col], mp[sno_col]):
        if scp in st.index and st.loc[scp, "diagnostic"] == 1:
            cls = st.loc[scp, "diagnostic_class"]
            if cls in SUPERCLASSES and pd.notna(sno):
                for code in str(sno).replace(";", ",").split(","):
                    code = code.strip().split(".")[0]
                    if code:
                        out[code] = cls
    print(f"{len(out)} SNOMED codes mapped to superclasses")
    return out


def map_snomed_to_superclass(codes, snomed_map):
    """Return the multi-hot superclass labels for a list of SNOMED CT codes."""
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for c in codes:
        cls = snomed_map.get(str(c).strip())
        if cls:
            y[SUPERCLASSES.index(cls)] = 1.0
    return y


def load_external(root, snomed_map, exclude_keyword="ptb", fs_out=100, seconds=10,
                  max_records=None):
    """Return (signals (n,12,1000), labels (n,5), info DataFrame)."""
    import wfdb
    headers = sorted(glob.glob(os.path.join(root, "**", "*.hea"), recursive=True))
    headers = [h for h in headers if exclude_keyword not in os.path.relpath(h, root).lower()]
    if max_records:
        headers = headers[:max_records]
    n_out = fs_out * seconds
    X, Y, info = [], [], []
    for h in headers:
        rec = h[:-4]
        try:
            sig, fields = wfdb.rdsamp(rec)
        except Exception as e:
            print("skip", rec, e)
            continue
        dx = []
        for line in fields.get("comments", []):
            if line.startswith("Dx:"):
                dx = line.split(":", 1)[1].strip().split(",")
        y = map_snomed_to_superclass(dx, snomed_map)
        if y.sum() == 0 or sig.shape[1] != 12:
            continue
        x = resample(np.nan_to_num(sig.T), fields["fs"], fs_out)[:, :n_out]
        if x.shape[1] < n_out:
            x = np.pad(x, ((0, 0), (0, n_out - x.shape[1])))
        X.append(x); Y.append(y)
        info.append({"record": os.path.relpath(rec, root),
                     "source": os.path.relpath(rec, root).split(os.sep)[0]})
    return np.stack(X).astype(np.float32), np.stack(Y), pd.DataFrame(info)
