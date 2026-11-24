# EEG-Based Seizure Detection Using SVM

> **Automated seizure detection from scalp EEG recordings using classical machine learning (SVM) with the CHB-MIT dataset.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **DISCLAIMER**: This is a research/educational project. It is **NOT** a clinical diagnostic system and should **NOT** be used for medical decision-making.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Interview Preparation Guide](#interview-preparation-guide)
- [References](#references)

---

## Overview

This project implements a complete machine learning pipeline for detecting epileptic seizures from scalp EEG recordings. The pipeline:

1. **Preprocesses** raw EDF-format EEG recordings (bandpass filtering, notch filtering, artifact rejection)
2. **Segments** continuous recordings into fixed-length windows (5s with 50% overlap)
3. **Extracts** interpretable time-domain and frequency-domain features
4. **Trains** three classifiers: Logistic Regression, Linear SVM, and RBF SVM
5. **Evaluates** using patient-wise splitting to prevent data leakage
6. **Generates** comprehensive metrics, visualizations, and per-patient analysis

---

## Dataset

**CHB-MIT Scalp EEG Database** ([PhysioNet](https://physionet.org/content/chbmit/1.0.0/) | [Kaggle](https://www.kaggle.com/datasets/borshalickson/chb-mit-data))

| Property | Value |
|----------|-------|
| Format | European Data Format (.edf) |
| Subjects | 22-23 pediatric patients |
| Channels | 23 EEG channels (10-20 system) |
| Sampling Rate | 256 Hz |
| Resolution | 16-bit |
| Total Duration | ~980+ hours of recording |
| Annotations | Seizure onset/offset times in summary files |

---

## Pipeline Architecture

```
Raw EDF Files
    │
    ▼
┌──────────────────────┐
│   1. Data Loading     │ Parse .edf files + summary annotations
│      & Inspection     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   2. Preprocessing    │ Bandpass (0.5-50 Hz) + Notch (60 Hz)
│                       │ + Amplitude clipping
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   3. Segmentation     │ 5-second windows, 50% overlap
│      & Labeling       │ Label: seizure ≥ 50% overlap
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   4. Feature          │ 22 features × 18 channels = 396 features
│      Extraction       │ Time + Frequency + Hjorth
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   5. Patient-wise     │ Train / Val / Test (no leakage)
│      Splitting        │ GroupKFold for CV
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   6. Model Training   │ LogReg │ Linear SVM │ RBF SVM
│      & Tuning         │ Balanced class weights
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   7. Evaluation &     │ Accuracy, Precision, Recall, F1,
│      Visualization    │ ROC-AUC, PR-AUC, Event Detection
└──────────────────────┘
```

---

## Features

### Extracted EEG Features (per channel, per window)

| Category | Features | Count |
|----------|----------|-------|
| **Time-domain** | Mean, Variance, Std, RMS, Skewness, Kurtosis, Line Length, Zero Crossings | 8 |
| **Band Powers** | Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-50) Hz | 5 |
| **Relative Band Powers** | Normalized band powers (fraction of total) | 5 |
| **Spectral** | Spectral Entropy | 1 |
| **Hjorth Parameters** | Activity, Mobility, Complexity | 3 |
| **Total per channel** | | **22** |

With 18 common EEG channels → **396 features per window**.

---

## Project Structure

```
EEG project/
├── config.yaml              # Pipeline configuration
├── requirements.txt          # Python dependencies
├── download_data.py          # Dataset download script
├── run_pipeline.py           # Main pipeline runner
├── .gitignore
├── README.md
│
├── src/                      # Source modules
│   ├── __init__.py
│   ├── utils.py              # Config loading, logging, helpers
│   ├── data_loader.py        # EDF loading, annotation parsing
│   ├── preprocessing.py      # Bandpass, notch filtering
│   ├── segmentation.py       # Window segmentation & labeling
│   ├── features.py           # Feature extraction
│   ├── models.py             # Training, tuning, splitting
│   ├── evaluation.py         # Metrics computation
│   └── visualization.py      # Figure generation
│
├── data/
│   ├── raw/                  # Raw CHB-MIT EDF files (not in git)
│   │   ├── chb01/
│   │   ├── chb02/
│   │   └── ...
│   └── processed/            # Extracted features
│       └── features.csv
│
├── models/                   # Saved trained models
│   ├── Logistic Regression.joblib
│   ├── Linear SVM.joblib
│   ├── RBF SVM.joblib
│   └── scaler.joblib
│
├── figures/                  # Generated visualizations
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── model_comparison.png
│   ├── class_distribution.png
│   ├── feature_importance_*.png
│   └── per_patient_f1_*.png
│
└── results/                  # Metrics and reports
    ├── dataset_stats.json
    ├── test_metrics.json
    ├── val_metrics.json
    ├── patient_split.json
    ├── event_detection.json
    └── per_patient_*.csv
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- ~25 GB free disk space (for the full dataset)

### 1. Clone and Install

```bash
git clone <repository-url>
cd "EEG project"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Kaggle API (recommended)**
```bash
# Set up Kaggle credentials first:
# https://www.kaggle.com/settings → API → Create New Token
# Place kaggle.json in C:\Users\<you>\.kaggle\

python download_data.py
```

**Option B: Manual Download**
1. Go to [CHB-MIT on Kaggle](https://www.kaggle.com/datasets/borshalickson/chb-mit-data)
2. Download and extract to `data/raw/`
3. Verify structure: `data/raw/chb01/`, `data/raw/chb02/`, etc.

### 3. Run Pipeline

```bash
python run_pipeline.py
```

### Optional Flags

```bash
python run_pipeline.py --skip-inspection     # Skip dataset inspection
python run_pipeline.py --load-features       # Reuse pre-extracted features
python run_pipeline.py --config my_config.yaml  # Custom config
```

---

## Results

> **Note**: Results below are placeholders. Run the pipeline to generate actual metrics from the dataset.

Results are saved to `results/` and include:
- `test_metrics.json` — All model metrics on the test set
- `val_metrics.json` — Validation set metrics
- `per_patient_*.csv` — Per-patient performance breakdown
- `event_detection.json` — Seizure event-level detection rates

Figures are saved to `figures/`:
- Confusion matrices, ROC curves, PR curves
- Feature importance analysis
- Per-patient F1 score bar charts
- Model comparison charts

---

## Technical Details

### Data Leakage Prevention

**Patient-wise splitting** ensures that all EEG windows from a given patient remain in the same data split (train, validation, or test). This prevents the model from memorizing patient-specific patterns and provides a realistic estimate of generalization to unseen patients.

### Class Imbalance Handling

Seizure events are extremely rare (~1-2% of total recording time). We address this through:
- **Balanced class weights**: `class_weight='balanced'` in SVM/LogReg, which inversely weights classes by their frequency
- **F1 score optimization**: Hyperparameter tuning maximizes F1 rather than accuracy
- **PR-AUC reporting**: More informative than ROC-AUC for imbalanced datasets

### Hyperparameter Tuning

- **GroupKFold** cross-validation ensures no patient appears in both train and validation folds during tuning
- **Linear SVM**: Tunes regularization parameter `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`
- **RBF SVM**: Jointly tunes `C` and `gamma` (kernel width)

---

## Interview Preparation Guide

### 1. Support Vector Machines (SVM)

**Q: What is an SVM and how does it work?**

An SVM finds the optimal hyperplane that maximizes the margin between two classes. The "support vectors" are the data points closest to the decision boundary. For linearly separable data, the SVM maximizes the geometric margin `2/||w||`. For non-separable data, soft-margin SVM allows some misclassifications controlled by the parameter `C`.

**Q: What are kernels and why are they used?**

Kernels implicitly map data to a higher-dimensional feature space where linear separation becomes possible (the "kernel trick"). Common kernels:
- **Linear**: `K(x,y) = x·y` — works when data is approximately linearly separable
- **RBF (Gaussian)**: `K(x,y) = exp(-γ||x-y||²)` — maps to infinite-dimensional space, handles non-linear boundaries
- **Polynomial**: `K(x,y) = (x·y + c)^d` — intermediate flexibility

### 2. Key Hyperparameters

**Q: What does C control?**

`C` is the regularization parameter that trades off between:
- **Small C**: Larger margin, more misclassifications allowed → better generalization, risk of underfitting
- **Large C**: Smaller margin, fewer misclassifications → fits training data closely, risk of overfitting

**Q: What does gamma (γ) control in RBF SVM?**

`γ` controls the "reach" of each support vector:
- **Small γ**: Each point influences a large area → smoother, more general decision boundary
- **Large γ**: Each point influences only nearby points → complex, potentially overfitting boundary

### 3. EEG Features

**Q: Why these specific features?**

- **Band powers**: Different brain rhythms (delta, theta, alpha, beta, gamma) have clinical significance. Seizures often show increased power in delta/theta bands and altered beta/gamma activity.
- **Hjorth parameters**: Capture signal complexity and frequency content without explicit spectral analysis.
- **Spectral entropy**: Measures the "disorder" of the power spectrum — seizures often show decreased entropy (more rhythmic activity).
- **Line length & RMS**: Sensitive to amplitude changes that characterize seizure onset.

### 4. Class Imbalance

**Q: Why is class imbalance a problem, and how do you handle it?**

In EEG seizure detection, seizure segments may be <2% of total data. A naive classifier that always predicts "non-seizure" would achieve >98% accuracy but 0% seizure detection (useless clinically).

Solutions used:
1. **Balanced class weights**: Weight the minority class loss higher (inversely proportional to class frequency)
2. **F1 score**: Harmonic mean of precision and recall — penalizes both false positives and false negatives
3. **PR-AUC**: Better metric than ROC-AUC for imbalanced datasets because it focuses on the minority class

### 5. Patient-Wise Splitting

**Q: Why not random train/test split?**

Random splitting would leak patient-specific information across splits. EEG signals have strong within-patient correlation (individual brain anatomy, impedance, resting rhythms). A model trained on some windows from Patient A and tested on other windows from Patient A would appear to perform well but actually relies on patient identity rather than seizure patterns.

**Patient-wise splitting** provides an honest estimate of how the model generalizes to **completely unseen patients**, which is the realistic deployment scenario.

### 6. Why SVM vs. Deep Learning?

- **Interpretability**: Feature importance is directly accessible (linear SVM coefficients)
- **Small data**: SVMs work well with limited training samples
- **Reproducibility**: Deterministic with fixed random seeds
- **Clinical acceptance**: Simpler models are easier to validate and explain

---

## References

1. Shoeb, A. H. (2009). *Application of machine learning to epileptic seizure onset detection and treatment*. PhD Thesis, MIT.
2. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.
3. CHB-MIT Scalp EEG Database: https://physionet.org/content/chbmit/1.0.0/
4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for educational and research purposes. Not for clinical use.*
