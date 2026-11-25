# Data

No data files are stored in this repository. Download each dataset into the folder shown, which Git ignores.

| Dataset | Folder | Source |
|---|---|---|
| PTB-XL | `data/raw/ptb-xl/` | https://physionet.org/content/ptb-xl/ |
| PTB-XL+ | `data/raw/ptb-xl-plus/` | https://physionet.org/content/ptb-xl-plus/ |
| EchoNext | `data/raw/echonext/` | https://physionet.org/content/echonext/ (credentialed access) |
| PhysioNet/CinC 2021 | `data/raw/cinc2021/` | https://physionet.org/content/challenge-2021/ |

Example for an open dataset:

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/ptb-xl https://physionet.org/files/ptb-xl/1.0.3/
```

EchoNext requires a PhysioNet credentialed account, the required training and a signed data use agreement. Its records must stay on your own machine or cloud storage and must not be committed or shared.

Processed arrays (filtered, normalised signals and fold indices) go in `data/processed/`, which is also ignored.
