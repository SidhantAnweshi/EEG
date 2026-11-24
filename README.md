# EEG Seizure Detection Pipeline

> **End-to-end machine learning system for automated epileptic seizure detection from scalp EEG recordings using the CHB-MIT dataset.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> ⚠️ **Disclaimer**: *This project is for research and educational purposes only. It is not intended for clinical diagnosis or medical decision-making.*

---

## Overview

This repository implements a modular, reproducible machine learning workflow for detecting epileptic seizures from multi-channel scalp EEG data (EDF format). It extracts interpretable time-domain, frequency-band, and spectral complexity features, applies patient-wise cross-validation to prevent data leakage, and benchmarks classical classifiers (Linear SVM, RBF SVM, Logistic Regression).

---

## Key Features

- **Signal Preprocessing**: Bandpass filtering (0.5–50 Hz), notch filtering (60 Hz powerline noise removal), and amplitude clipping to suppress artifacts.
- **Window Segmentation**: Continuous signal segmenting into 5-second sliding windows with 50% overlap; seizure events labeled based on temporal overlap thresholds.
- **Comprehensive Feature Extraction**: 22 features per channel (396 total across 18 standardized channels):
  - *Time-domain*: Mean, Variance, Std, RMS, Skewness, Kurtosis, Line Length, Zero Crossings.
  - *Frequency-domain*: Absolute and relative band powers (Delta, Theta, Alpha, Beta, Gamma).
  - *Complexity & Dynamics*: Spectral entropy and Hjorth parameters (Activity, Mobility, Complexity).
- **Leakage-Free Evaluation**: Patient-wise partitioning ensures models are evaluated on unseen subjects rather than memorizing patient-specific baselines.
- **Imbalance Handling**: Class-weighted cost functions and F1 / PR-AUC optimization to handle severe class imbalance (~1-2% seizure prevalence).

---

## Project Structure

```
├── config.yaml            # Pipeline hyperparameters & paths
├── download_data.py       # Automated data downloader (Kaggle / PhysioNet)
├── run_pipeline.py        # Main execution entrypoint
├── requirements.txt       # Dependencies
├── src/                   # Source modules
│   ├── data_loader.py     # EDF parser & metadata inspector
│   ├── preprocessing.py   # Filtering & signal cleaning
│   ├── segmentation.py    # Sliding window & labeling
│   ├── features.py        # Feature extraction engine
│   ├── models.py          # Classifiers & hyperparameter tuning
│   ├── evaluation.py      # Metrics calculation & event detection
│   └── visualization.py   # Publication-ready plots
├── figures/               # Generated evaluation plots & curves
└── results/               # Metrics JSONs & per-patient CSVs
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/SidhantAnweshi/EEG.git
cd EEG

# Set up virtual environment
python -m venv venv
venv\Scripts\activate      # Windows (or source venv/bin/activate on Linux/macOS)

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the **CHB-MIT Scalp EEG Database** via Kaggle API:
```bash
# Place your kaggle.json in ~/.kaggle/ (or C:\Users\<Username>\.kaggle\)
python download_data.py
```
*(Alternatively, place extracted patient folders like `chb01/`, `chb02/` directly into `data/raw/`.)*

### 3. Run Pipeline

```bash
# Execute end-to-end (inspect, preprocess, extract, train, evaluate)
python run_pipeline.py

# Re-use already extracted features to save time
python run_pipeline.py --load-features
```

---

## Results & Visualizations

Output metrics are saved in `results/` (`test_metrics.json`, `val_metrics.json`, `per_patient_*.csv`).

Visualizations are saved directly to `figures/`:
- **Model Comparison**: `figures/model_comparison.png`
- **Confusion Matrices**: `figures/confusion_matrices.png`
- **ROC & PR Curves**: `figures/roc_curves.png`, `figures/pr_curves.png`
- **Feature Importance**: `figures/feature_importance_*.png`
- **Patient Generalization**: `figures/per_patient_f1_*.png`

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
