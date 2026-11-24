"""
EEG Seizure Detection Pipeline — Main Runner

Complete pipeline for seizure detection from the CHB-MIT Scalp EEG dataset:
    1. Dataset inspection
    2. EEG preprocessing (bandpass + notch filtering)
    3. Fixed-window segmentation with seizure labeling
    4. Feature extraction (time/frequency domain)
    5. Patient-wise train/val/test split (no data leakage)
    6. Feature scaling
    7. Model training: Logistic Regression, Linear SVM, RBF SVM
    8. Group-aware hyperparameter tuning
    9. Comprehensive evaluation
    10. Visualization
    11. Model & results persistence

Usage:
    python run_pipeline.py
    python run_pipeline.py --config config.yaml

DISCLAIMER: This is a research/educational project. It is NOT a clinical
diagnostic system and should NOT be used for medical decisions.
"""

import os
import sys
import argparse
import time
import logging
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, ensure_dirs, save_json
from src.data_loader import (
    discover_patients,
    get_patient_files,
    get_summary_path,
    parse_summary_file,
    load_edf_file,
    inspect_dataset,
)
from src.preprocessing import preprocess_eeg
from src.segmentation import segment_and_label_file
from src.features import extract_window_features, get_feature_names
from src.models import (
    patient_wise_split,
    prepare_split_data,
    scale_features,
    train_logistic_regression,
    train_linear_svm,
    train_rbf_svm,
    save_models,
)
from src.evaluation import (
    evaluate_model,
    per_patient_evaluation,
    seizure_event_detection,
)
from src.visualization import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Seizure Detection Pipeline (CHB-MIT + SVM)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-inspection", action="store_true",
        help="Skip dataset inspection step"
    )
    parser.add_argument(
        "--load-features", action="store_true",
        help="Load pre-extracted features instead of reprocessing"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate and use synthetic sample EEG data for testing"
    )
    return parser.parse_args()


def step_1_inspect(config, logger, use_synthetic=False):
    """Step 1: Inspect the dataset structure."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATASET INSPECTION")
    logger.info("=" * 70)
    
    data_dir = config["paths"]["raw_data"]
    patient_subset = config["dataset"].get("patients_subset", None)
    
    # Discover available patients
    all_patients = discover_patients(data_dir)
    
    if not all_patients or use_synthetic:
        if use_synthetic or not all_patients:
            logger.info("No raw patient EDF files found (or --synthetic specified).")
            logger.info("Generating synthetic EEG dataset for full pipeline execution & testing...")
            from src.synthetic import generate_synthetic_dataset
            generate_synthetic_dataset(data_dir=data_dir, patients=patient_subset)
            all_patients = discover_patients(data_dir)
    
    # Use subset if configured, otherwise use all
    if patient_subset:
        patients = [p for p in patient_subset if p in all_patients]
        if not patients:
            logger.warning(
                f"None of the configured patients found. Using all available."
            )
            patients = all_patients
    else:
        patients = all_patients
    
    logger.info(f"\nProcessing patients: {patients}")
    
    # Detailed inspection
    stats = inspect_dataset(data_dir, patients)
    
    # Save stats
    stats_path = os.path.join(config["paths"]["results"], "dataset_stats.json")
    save_json(stats, stats_path)
    logger.info(f"\nDataset statistics saved to {stats_path}")
    
    return patients, stats


def step_2_process_and_extract(config, patients, logger):
    """Step 2-4: Preprocess, segment, and extract features."""
    logger.info("\n" + "=" * 70)
    logger.info("STEPS 2-4: PREPROCESSING → SEGMENTATION → FEATURE EXTRACTION")
    logger.info("=" * 70)
    
    data_dir = config["paths"]["raw_data"]
    common_channels = config["dataset"]["common_channels"]
    fs = config["dataset"]["sampling_rate"]
    freq_bands = {
        name: tuple(band) for name, band in config["features"]["freq_bands"].items()
    }
    
    all_features = []
    all_labels = []
    all_patient_ids = []
    all_file_names = []
    all_window_indices = []
    
    feature_names = None
    total_windows = 0
    total_seizure_windows = 0
    
    for patient_id in patients:
        logger.info(f"\nProcessing {patient_id}...")
        t_start = time.time()
        
        # Get files and annotations
        edf_files = get_patient_files(data_dir, patient_id)
        summary_path = get_summary_path(data_dir, patient_id)
        
        if summary_path:
            seizure_annotations = parse_summary_file(summary_path)
        else:
            seizure_annotations = {}
        
        patient_windows = 0
        patient_seizure_windows = 0
        
        for edf_path in edf_files:
            filename = os.path.basename(edf_path)
            
            # Get seizure intervals for this file
            seizure_intervals = seizure_annotations.get(filename, [])
            
            # Load EDF
            data, ch_names, sfreq = load_edf_file(
                edf_path, common_channels=common_channels, verbose=False
            )
            
            if data is None:
                logger.warning(f"  Skipping {filename} (failed to load)")
                continue
            
            # Check sampling rate
            if abs(sfreq - fs) > 1:
                logger.warning(
                    f"  Skipping {filename}: expected {fs}Hz, got {sfreq}Hz"
                )
                continue
            
            # Initialize feature names on first successful file
            if feature_names is None:
                feature_names = get_feature_names(ch_names, freq_bands)
            
            # Preprocess
            data = preprocess_eeg(data, config)
            
            # Segment and label
            windows, labels, positions = segment_and_label_file(
                data, sfreq, seizure_intervals, config
            )
            
            if not windows:
                continue
            
            # Extract features from each window
            for w_idx, (window, label) in enumerate(zip(windows, labels)):
                try:
                    feats = extract_window_features(window, sfreq, freq_bands)
                    
                    # Check for NaN/Inf
                    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    all_features.append(feats)
                    all_labels.append(label)
                    all_patient_ids.append(patient_id)
                    all_file_names.append(filename)
                    all_window_indices.append(w_idx)
                    
                except Exception as e:
                    logger.debug(f"  Feature extraction failed for window {w_idx}: {e}")
                    continue
            
            n_seizure_in_file = int(np.sum(labels))
            patient_windows += len(windows)
            patient_seizure_windows += n_seizure_in_file
        
        elapsed = time.time() - t_start
        logger.info(
            f"  {patient_id}: {patient_windows} windows "
            f"({patient_seizure_windows} seizure) in {elapsed:.1f}s"
        )
        
        total_windows += patient_windows
        total_seizure_windows += patient_seizure_windows
    
    # Build DataFrame
    logger.info(f"\nTotal: {total_windows} windows, {total_seizure_windows} seizure windows")
    
    if not all_features:
        logger.error("No features extracted! Check data directory and file formats.")
        sys.exit(1)
    
    X = np.array(all_features)
    
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df["label"] = all_labels
    features_df["patient_id"] = all_patient_ids
    features_df["file"] = all_file_names
    features_df["window_idx"] = all_window_indices
    
    # Save features
    features_path = os.path.join(config["paths"]["processed_data"], "features.csv")
    features_df.to_csv(features_path, index=False)
    logger.info(f"Features saved to {features_path}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature names ({len(feature_names)}): {feature_names[:5]}...")
    
    return features_df, feature_names


def step_5_split_and_train(config, features_df, feature_names, logger):
    """Step 5-8: Split data, train models, tune hyperparameters."""
    logger.info("\n" + "=" * 70)
    logger.info("STEPS 5-8: SPLITTING → TRAINING → TUNING")
    logger.info("=" * 70)
    
    # Patient-wise split
    patient_ids = features_df["patient_id"].unique().tolist()
    split_config = config["splitting"]
    
    patient_split = patient_wise_split(
        patient_ids,
        test_size=split_config["test_size"],
        val_size=split_config["val_size"],
        random_state=split_config["random_state"],
    )
    
    # Save split info
    save_json(patient_split, os.path.join(config["paths"]["results"], "patient_split.json"))
    
    # Prepare data
    split_data, feature_cols = prepare_split_data(features_df, patient_split)
    
    X_train, y_train, groups_train = split_data["train"]
    X_val, y_val, groups_val = split_data["val"]
    X_test, y_test, groups_test = split_data["test"]
    
    # Log class distribution
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Train: {np.sum(y_train==0)} non-seizure, {np.sum(y_train==1)} seizure "
                f"(ratio: 1:{np.sum(y_train==0)/max(1,np.sum(y_train==1)):.0f})")
    logger.info(f"  Val:   {np.sum(y_val==0)} non-seizure, {np.sum(y_val==1)} seizure")
    logger.info(f"  Test:  {np.sum(y_test==0)} non-seizure, {np.sum(y_test==1)} seizure")
    
    # Scale features (fit on train only!)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
    
    # ---- Train Models ----
    
    logger.info("\n--- Training Models ---")
    
    # 1. Logistic Regression (Baseline)
    t0 = time.time()
    lr_model = train_logistic_regression(X_train_s, y_train, config)
    logger.info(f"  LR training time: {time.time()-t0:.1f}s")
    
    # 2. Linear SVM with group-aware tuning
    t0 = time.time()
    linear_svm = train_linear_svm(X_train_s, y_train, groups_train, config)
    logger.info(f"  Linear SVM training time: {time.time()-t0:.1f}s")
    
    # 3. RBF SVM with group-aware tuning
    t0 = time.time()
    rbf_svm = train_rbf_svm(X_train_s, y_train, groups_train, config)
    logger.info(f"  RBF SVM training time: {time.time()-t0:.1f}s")
    
    models = {
        "Logistic Regression": lr_model,
        "Linear SVM": linear_svm,
        "RBF SVM": rbf_svm,
    }
    
    # Save models
    save_models(models, scaler, config["paths"]["models"])
    
    return models, scaler, split_data, feature_cols, patient_split


def step_6_evaluate(config, models, scaler, split_data, feature_cols, 
                    features_df, feature_names, patient_split, logger):
    """Step 9-10: Evaluate models and generate visualizations."""
    logger.info("\n" + "=" * 70)
    logger.info("STEPS 9-10: EVALUATION → VISUALIZATION")
    logger.info("=" * 70)
    
    X_train, y_train, _ = split_data["train"]
    X_val, y_val, _ = split_data["val"]
    X_test, y_test, _ = split_data["test"]
    
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)
    
    # ---- Evaluate on Test Set ----
    logger.info("\n--- Test Set Evaluation ---")
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test_s, y_test, model_name)
        all_metrics[model_name] = metrics
    
    # ---- Validation Set Evaluation ----
    logger.info("\n--- Validation Set Evaluation ---")
    val_metrics = {}
    for model_name, model in models.items():
        v_metrics = evaluate_model(model, X_val_s, y_val, f"{model_name} (Val)")
        val_metrics[model_name] = v_metrics
    
    # ---- Per-Patient Evaluation ----
    logger.info("\n--- Per-Patient Evaluation ---")
    patient_results = {}
    for model_name, model in models.items():
        df_results = per_patient_evaluation(
            model, features_df, feature_cols, scaler, model_name
        )
        patient_results[model_name] = df_results
    
    # ---- Seizure Event Detection ----
    logger.info("\n--- Seizure Event Detection ---")
    event_results = {}
    for model_name, model in models.items():
        events = seizure_event_detection(
            model, features_df, feature_cols, scaler,
            window_seconds=config["segmentation"]["window_seconds"],
            model_name=model_name
        )
        event_results[model_name] = events
    
    # ---- Save Results ----
    results_dir = config["paths"]["results"]
    
    # Save test metrics
    save_json(
        {k: {mk: mv for mk, mv in v.items() 
             if mk not in ["roc_curve", "pr_curve"]}
         for k, v in all_metrics.items()},
        os.path.join(results_dir, "test_metrics.json")
    )
    
    # Save validation metrics
    save_json(
        {k: {mk: mv for mk, mv in v.items() 
             if mk not in ["roc_curve", "pr_curve"]}
         for k, v in val_metrics.items()},
        os.path.join(results_dir, "val_metrics.json")
    )
    
    # Save per-patient results
    for model_name, df in patient_results.items():
        safe_name = model_name.lower().replace(" ", "_")
        df.to_csv(
            os.path.join(results_dir, f"per_patient_{safe_name}.csv"), index=False
        )
    
    # Save event results
    save_json(
        {k: {ek: ev for ek, ev in v.items() if ek != "details"}
         for k, v in event_results.items()},
        os.path.join(results_dir, "event_detection.json")
    )
    
    # ---- Generate Figures ----
    generate_all_figures(
        all_metrics, features_df, models, feature_names,
        patient_results, config["paths"]["figures"]
    )
    
    return all_metrics, val_metrics, patient_results, event_results


def print_summary(all_metrics, event_results, logger):
    """Print final summary of results."""
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    # Find best model by F1
    best_model = max(all_metrics, key=lambda k: all_metrics[k].get("f1", 0))
    best_f1 = all_metrics[best_model]["f1"]
    
    logger.info(f"\nBest Model: {best_model} (F1={best_f1:.4f})")
    
    logger.info("\n┌─────────────────────┬──────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    logger.info("│ Model               │ Accuracy │ Precision │ Recall   │ Specific │ F1       │ ROC-AUC  │ PR-AUC   │")
    logger.info("├─────────────────────┼──────────┼───────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    
    for model_name, metrics in all_metrics.items():
        logger.info(
            f"│ {model_name:<19} │ "
            f"{metrics['accuracy']:.4f}   │ "
            f"{metrics['precision']:.4f}    │ "
            f"{metrics['recall']:.4f}   │ "
            f"{metrics['specificity']:.4f}   │ "
            f"{metrics['f1']:.4f}   │ "
            f"{metrics['roc_auc']:.4f}   │ "
            f"{metrics['pr_auc']:.4f}   │"
        )
    
    logger.info("└─────────────────────┴──────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    
    # Event detection summary
    logger.info("\nSeizure Event Detection:")
    for model_name, events in event_results.items():
        logger.info(
            f"  {model_name}: "
            f"{events['detected_events']}/{events['total_events']} events detected "
            f"({events['detection_rate']:.1%})"
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("DISCLAIMER: This is a research/educational project.")
    logger.info("It is NOT a clinical diagnostic system.")
    logger.info("=" * 70)


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging()
    
    # Create directories
    ensure_dirs(config)
    
    logger.info("=" * 70)
    logger.info("EEG SEIZURE DETECTION PIPELINE")
    logger.info("Dataset: CHB-MIT Scalp EEG")
    logger.info("Models: Logistic Regression, Linear SVM, RBF SVM")
    logger.info("=" * 70)
    
    pipeline_start = time.time()
    
    # ---- Step 1: Inspect Dataset ----
    if not args.skip_inspection:
        patients, stats = step_1_inspect(config, logger, use_synthetic=args.synthetic)
    else:
        data_dir = config["paths"]["raw_data"]
        patients = config["dataset"].get("patients_subset", discover_patients(data_dir))
        stats = None
    
    # ---- Steps 2-4: Process and Extract Features ----
    if args.load_features:
        features_path = os.path.join(config["paths"]["processed_data"], "features.csv")
        if os.path.exists(features_path):
            logger.info(f"\nLoading pre-extracted features from {features_path}")
            features_df = pd.read_csv(features_path)
            feature_cols = [
                col for col in features_df.columns
                if col not in ['patient_id', 'label', 'file', 'window_idx']
            ]
            feature_names = feature_cols
        else:
            logger.warning("Features file not found, reprocessing...")
            features_df, feature_names = step_2_process_and_extract(config, patients, logger)
    else:
        features_df, feature_names = step_2_process_and_extract(config, patients, logger)
    
    # ---- Steps 5-8: Split, Train, Tune ----
    models, scaler, split_data, feature_cols, patient_split = step_5_split_and_train(
        config, features_df, feature_names, logger
    )
    
    # ---- Steps 9-10: Evaluate and Visualize ----
    all_metrics, val_metrics, patient_results, event_results = step_6_evaluate(
        config, models, scaler, split_data, feature_cols,
        features_df, feature_names, patient_split, logger
    )
    
    # ---- Final Summary ----
    print_summary(all_metrics, event_results, logger)
    
    elapsed = time.time() - pipeline_start
    logger.info(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")
    logger.info("All outputs saved to: models/, figures/, results/")


if __name__ == "__main__":
    main()
