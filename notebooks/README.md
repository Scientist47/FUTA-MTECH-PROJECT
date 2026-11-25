# Notebooks

Run these in Google Colab (or Kaggle) in order. Each one starts with the same setup cell, which connects Google Drive, clones this repository and installs the libraries. Data and results are saved in `MyDrive/FUTA-MTECH/`, never in this repository.

| Notebook | Roadmap step | Thesis sections | Chapter 4 outputs |
|---|---|---|---|
| `01_setup_and_explore_ptbxl.ipynb` | Set up, download PTB-XL and PTB-XL+ | 3.2.1-3.2.2 | Table 4.1, Figure 4.1 |
| `02_preprocess_ptbxl.ipynb` | Filter, split by folds, z-score | 3.4.1-3.4.2 | Figure 4.2 |
| `03_train_and_evaluate_models.ipynb` | 1D-CNN and CNN-BiLSTM-attention x 5 seeds (Eq. 1), focal-loss check, EBM, test-fold evaluation, external validation | 3.3, 3.4.3, 3.5.1-3.5.2 | Tables 4.2-4.5, Figure 4.3 |
| `04_explanations.ipynb` | SHAP, Grad-CAM, attention; segment-by-lead grid (Eq. 2) | 3.3.5 | Figures 4.4-4.5, Table 4.6 |
| `05_faithfulness.ipynb` | Deletion/insertion (Eq. 3), randomisation, stability, agreement, known-answer tests | 3.5.3 | Tables 4.7-4.11, Figure 4.6 |
| `06_clinician_study.ipynb` | Blinded rating booklet and forms; Δ (Eq. 4), agreement, Jaccard | 3.5.4 | Table 4.12, Figure 4.7 |
| `07_echonext_extension.ipynb` | Retrain the 1D-CNN for SHD on EchoNext, explain and test | 3.2.3, 3.4.4 | Tables 4.13-4.16, Figure 4.8 |

Open one in Colab: **File > Open notebook > GitHub**, type `Scientist47/FUTA-MTECH-PROJECT`, pick the notebook, then **File > Save a copy in Drive**.

Set `QUICK_TEST` (environment variable `FUTA_QUICK_TEST=1`) only to check that a notebook runs; it uses tiny settings. Clear outputs that show patient-level data before committing a notebook.
