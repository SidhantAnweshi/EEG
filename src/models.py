"""
Model Training Module.

Implements:
- Patient-wise data splitting (prevents data leakage)
- Logistic Regression baseline
- Linear SVM
- RBF SVM
- Group-aware hyperparameter tuning with GroupKFold
- Class imbalance handling via balanced class weights
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

logger = logging.getLogger("EEG-Pipeline")


def patient_wise_split(patient_ids, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split patients into train/validation/test sets.
    
    CRITICAL: This ensures no data leakage — all windows from a given 
    patient stay in the same split.
    
    Args:
        patient_ids: List of unique patient IDs
        test_size: Fraction of patients for test set
        val_size: Fraction of patients for validation set
        random_state: Random seed
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]} patient ID lists
    """
    rng = np.random.RandomState(random_state)
    
    # Shuffle patients
    patients = list(patient_ids)
    rng.shuffle(patients)
    
    n = len(patients)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    n_train = n - n_test - n_val
    
    if n_train < 1:
        logger.warning("Very few patients — adjusting split sizes")
        n_train = max(1, n - 2)
        n_val = min(1, n - n_train)
        n_test = n - n_train - n_val
    
    split = {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }
    
    logger.info(f"Patient-wise split:")
    logger.info(f"  Train ({len(split['train'])} patients): {split['train']}")
    logger.info(f"  Val   ({len(split['val'])} patients): {split['val']}")
    logger.info(f"  Test  ({len(split['test'])} patients): {split['test']}")
    
    return split


def prepare_split_data(features_df, patient_split):
    """
    Split the feature DataFrame according to patient-wise split.
    
    Args:
        features_df: DataFrame with columns including 'patient_id', 'label', 
                     and feature columns
        patient_split: dict from patient_wise_split()
    
    Returns:
        dict: {'train': (X, y, groups), 'val': (X, y, groups), 'test': (X, y, groups)}
    """
    feature_cols = [
        col for col in features_df.columns 
        if col not in ['patient_id', 'label', 'file', 'window_idx']
    ]
    
    split_data = {}
    for split_name, patients in patient_split.items():
        mask = features_df['patient_id'].isin(patients)
        df_split = features_df[mask]
        
        X = df_split[feature_cols].values
        y = df_split['label'].values
        groups = df_split['patient_id'].values
        
        logger.info(
            f"  {split_name}: {len(X)} samples, "
            f"{np.sum(y == 1)} seizure, {np.sum(y == 0)} non-seizure"
        )
        
        split_data[split_name] = (X, y, groups)
    
    return split_data, feature_cols


def scale_features(X_train, X_val, X_test):
    """
    Standardize features using training set statistics only.
    
    IMPORTANT: Fit scaler on training data only to prevent data leakage.
    
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Replace any NaN/Inf from scaling
    for X in [X_train_scaled, X_val_scaled, X_test_scaled]:
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, config):
    """Train Logistic Regression baseline with balanced class weights."""
    lr_config = config["models"]["logistic_regression"]
    
    model = LogisticRegression(
        max_iter=lr_config["max_iter"],
        solver=lr_config["solver"],
        class_weight="balanced",
        random_state=config["random_state"],
        n_jobs=-1,
    )
    
    logger.info("Training Logistic Regression (balanced class weights)...")
    model.fit(X_train, y_train)
    logger.info("  Logistic Regression training complete.")
    
    return model


def train_linear_svm(X_train, y_train, groups_train, config):
    """
    Train Linear SVM with group-aware hyperparameter tuning.
    
    Uses GroupKFold to ensure patient-wise cross-validation during tuning.
    """
    svm_config = config["models"]["linear_svm"]
    cv_config = config["cross_validation"]
    
    # Build parameter grid
    param_grid = {"C": svm_config["C_range"]}
    
    # Group-aware CV
    n_unique_groups = len(np.unique(groups_train))
    n_splits = min(cv_config["n_splits"], n_unique_groups)
    
    if n_splits < 2:
        logger.warning("Not enough groups for GroupKFold CV — training with default C=1")
        model = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            random_state=config["random_state"],
            probability=True,
        )
        model.fit(X_train, y_train)
        return model
    
    group_kfold = GroupKFold(n_splits=n_splits)
    
    base_model = SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=config["random_state"],
        probability=True,
    )
    
    scorer = make_scorer(f1_score, zero_division=0)
    
    logger.info(f"Tuning Linear SVM (GroupKFold, {n_splits} splits)...")
    logger.info(f"  C values: {svm_config['C_range']}")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=group_kfold,
        scoring=scorer,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    logger.info(f"  Best C: {grid_search.best_params_['C']}")
    logger.info(f"  Best CV F1: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_rbf_svm(X_train, y_train, groups_train, config):
    """
    Train RBF SVM with group-aware hyperparameter tuning.
    
    Tunes both C and gamma using GroupKFold cross-validation.
    """
    svm_config = config["models"]["rbf_svm"]
    cv_config = config["cross_validation"]
    
    # Build parameter grid
    param_grid = {
        "C": svm_config["C_range"],
        "gamma": svm_config["gamma_range"],
    }
    
    # Group-aware CV
    n_unique_groups = len(np.unique(groups_train))
    n_splits = min(cv_config["n_splits"], n_unique_groups)
    
    if n_splits < 2:
        logger.warning("Not enough groups for GroupKFold CV — training with defaults")
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=config["random_state"],
            probability=True,
        )
        model.fit(X_train, y_train)
        return model
    
    group_kfold = GroupKFold(n_splits=n_splits)
    
    base_model = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=config["random_state"],
        probability=True,
    )
    
    scorer = make_scorer(f1_score, zero_division=0)
    
    logger.info(f"Tuning RBF SVM (GroupKFold, {n_splits} splits)...")
    logger.info(f"  C values: {svm_config['C_range']}")
    logger.info(f"  gamma values: {svm_config['gamma_range']}")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=group_kfold,
        scoring=scorer,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    logger.info(f"  Best params: {grid_search.best_params_}")
    logger.info(f"  Best CV F1: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def save_models(models, scaler, model_dir):
    """Save trained models and scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    
    for name, model in models.items():
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(model, path)
        logger.info(f"  Saved model: {path}")
    
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Saved scaler: {scaler_path}")
