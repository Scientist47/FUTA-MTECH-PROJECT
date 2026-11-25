# Explainable Deep Learning for 12-Lead ECG Abnormality Classification

Code for the MTech thesis **"Development of an Explainable Deep Learning Model for 12-Lead Electrocardiogram Abnormality Classification"**, Department of Computer Science, Federal University of Technology, Akure (FUTA).

- **Author:** Adewuyi Olatunde Oke
- **Supervisor:** Dr Akintoba Akinwonmi
- **Status:** code for every step of the methodology is in place and runs end to end on a small synthetic sample. No results are reported here yet; they come from runs on the real datasets.

## Aim and objectives

The aim of the study is to develop an explainable deep learning model for multi-label classification of 12-lead ECGs, and to test whether its explanations are faithful to the model and clinically plausible to cardiologists. The specific objectives are to:

| Objective | What it covers | Where it lives in this repository |
|---|---|---|
| **(a) Design** an explainable deep learning model | Residual 1D-CNN and CNN-BiLSTM with attention; class-weighted binary cross-entropy (Eq. 1); SHAP, Grad-CAM and attention explanations aligned to a 5-segment x 12-lead grid (Eq. 2); Explainable Boosting Machine (EBM) comparator | `models/`, `xai/` |
| **(b) Implement** the model in (a) on PTB-XL and extend it to EchoNext | Preprocessing (100 Hz, 0.5 Hz high-pass, 50 Hz notch, per-lead z-score), patient-level folds (train 1-8, validate 9, test 10), five random seeds; retraining the 1D-CNN on EchoNext for structural heart disease | `data/`, `preprocessing/`, `training/` |
| **(c) Evaluate** the model in (b) | Classification performance (macro-AUROC, F1, calibration) and external validation on PhysioNet/CinC 2021; faithfulness of the explanations (deletion and insertion AUC, Eq. 3, plus randomisation, stability, agreement and known-answer tests); clinical plausibility in a blinded cardiologist rating study (Eq. 4) | `evaluation/` |

## Repository layout

```
data/            How to obtain each dataset. Data files are never committed.
preprocessing/   Reading records, filtering, normalisation, fold assignment     (Objective b)
models/          1D-CNN, CNN-BiLSTM with attention, loss, EBM comparator        (Objective a)
xai/             SHAP, Grad-CAM, attention extraction, segment-by-lead grid     (Objective a)
training/        Training loop, seeds and experiment configs                    (Objective b)
evaluation/      Performance, external validation, faithfulness, plausibility   (Objective c)
notebooks/       Exploration and figure notebooks
results/         Generated metrics, figures and tables (not committed)
```

The step-by-step workflow is in the seven numbered notebooks in [`notebooks/`](notebooks/README.md), which import the code in the folders above.

## Datasets

| Dataset | Use in the study | Access |
|---|---|---|
| [PTB-XL](https://physionet.org/content/ptb-xl/) (Wagner et al., 2020) | Primary dataset, five diagnostic superclasses (NORM, MI, STTC, CD, HYP) | Open, CC BY 4.0 |
| [PTB-XL+](https://physionet.org/content/ptb-xl-plus/) (Strodthoff et al., 2023) | Measured features for the EBM comparator; fiducial points for segment alignment | Open, CC BY 4.0 |
| [EchoNext](https://physionet.org/content/echonext/) (Elias & Finer, 2025) | Extension to echocardiogram-confirmed structural heart disease | PhysioNet credentialed access and data use agreement |
| [PhysioNet/CinC Challenge 2021](https://physionet.org/content/challenge-2021/) (Reyna et al., 2021) | External validation (PTB and PTB-XL sources excluded) | Open |

See [`data/README.md`](data/README.md) for download steps. EchoNext data must not be shared or committed to this repository under its data use agreement.

## Getting started

```bash
git clone https://github.com/Scientist47/FUTA-MTECH-PROJECT.git
cd FUTA-MTECH-PROJECT
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main libraries: PyTorch, wfdb, shap, Captum, InterpretML, scikit-learn and NeuroKit2. Tested versions are listed in `requirements.txt`.

The experiments are planned to run on free or low-cost cloud GPUs (Google Colab or Kaggle, NVIDIA T4) as well as a laptop.

## Reproducibility

- Fixed random seeds; each architecture is trained with five seeds and reported as a mean with 95% confidence intervals.
- All preprocessing statistics are fitted on PTB-XL folds 1-8 only.
- Fold 10 (the test fold) is used once, after the hypotheses and analysis plan have been fixed.
- Pinned software environment.

## Ethics

PTB-XL, PTB-XL+ and the CinC 2021 data are de-identified and openly available. EchoNext is used under PhysioNet credentialed access. The clinician rating study is carried out only after approval from the University Health Research Ethics Committee, with written informed consent; no rater data are stored in this repository. This code is for research and is not a tool for clinical use.

## Licence

MIT. See [LICENSE](LICENSE). The datasets are covered by their own licences.
