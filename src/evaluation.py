"""
Evaluation Module.

Computes comprehensive metrics for seizure detection:
- Accuracy, Precision, Recall/Sensitivity, Specificity
- F1 Score, ROC-AUC, PR-AUC
- Per-patient evaluation
- Seizure-event-level detection analysis
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger("EEG-Pipeline")


def compute_specificity(y_true, y_pred):
    """Compute specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)


def evaluate_model(model, X, y, model_name="Model"):
    """
    Compute all evaluation metrics for a model.
    
    Args:
        model: Trained classifier with predict and predict_proba methods
        X: Feature matrix
        y: True labels
        model_name: Name for logging
    
    Returns:
        dict: All computed metrics
    """
    y_pred = model.predict(X)
    
    # Get probability scores for positive class
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = model.decision_function(X)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "specificity": compute_specificity(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
    
    # ROC-AUC and PR-AUC (require both classes present)
    if len(np.unique(y)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y, y_prob)
        except Exception:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = average_precision_score(y, y_prob)
        except Exception:
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    
    # ROC and PR curve data
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        prec, rec, _ = precision_recall_curve(y, y_prob)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
    
    # Log results
    logger.info(f"\n  {model_name} Results:")
    logger.info(f"    Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"    Precision:   {metrics['precision']:.4f}")
    logger.info(f"    Recall:      {metrics['recall']:.4f}")
    logger.info(f"    Specificity: {metrics['specificity']:.4f}")
    logger.info(f"    F1:          {metrics['f1']:.4f}")
    logger.info(f"    ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"    PR-AUC:      {metrics['pr_auc']:.4f}")
    logger.info(f"    Confusion Matrix:\n      {cm}")
    
    return metrics


def per_patient_evaluation(model, features_df, feature_cols, scaler, model_name="Model"):
    """
    Evaluate model performance per patient.
    
    Args:
        model: Trained classifier
        features_df: Full feature DataFrame
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler
        model_name: Name for logging
    
    Returns:
        DataFrame: Per-patient metrics
    """
    patient_results = []
    
    for patient_id in sorted(features_df['patient_id'].unique()):
        mask = features_df['patient_id'] == patient_id
        df_patient = features_df[mask]
        
        X_patient = scaler.transform(df_patient[feature_cols].values)
        X_patient[np.isnan(X_patient)] = 0
        X_patient[np.isinf(X_patient)] = 0
        y_patient = df_patient['label'].values
        
        if len(np.unique(y_patient)) < 2:
            # Patient has only one class (likely all non-seizure)
            y_pred = model.predict(X_patient)
            acc = accuracy_score(y_patient, y_pred)
            patient_results.append({
                "patient_id": patient_id,
                "n_samples": len(y_patient),
                "n_seizure": int(np.sum(y_patient == 1)),
                "n_normal": int(np.sum(y_patient == 0)),
                "accuracy": acc,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "note": "single_class"
            })
            continue
        
        y_pred = model.predict(X_patient)
        
        patient_results.append({
            "patient_id": patient_id,
            "n_samples": len(y_patient),
            "n_seizure": int(np.sum(y_patient == 1)),
            "n_normal": int(np.sum(y_patient == 0)),
            "accuracy": accuracy_score(y_patient, y_pred),
            "precision": precision_score(y_patient, y_pred, zero_division=0),
            "recall": recall_score(y_patient, y_pred, zero_division=0),
            "f1": f1_score(y_patient, y_pred, zero_division=0),
            "note": ""
        })
    
    df_results = pd.DataFrame(patient_results)
    
    logger.info(f"\n  Per-Patient Results ({model_name}):")
    logger.info(f"\n{df_results.to_string(index=False)}")
    
    return df_results


def seizure_event_detection(model, features_df, feature_cols, scaler, 
                             window_seconds=5, model_name="Model"):
    """
    Evaluate seizure detection at the event level.
    
    A seizure event is considered "detected" if at least one window within
    the seizure period is correctly classified as seizure.
    
    Returns:
        dict: Event-level detection metrics
    """
    results = {
        "total_events": 0,
        "detected_events": 0,
        "missed_events": 0,
        "detection_rate": 0.0,
        "details": [],
    }
    
    for patient_id in sorted(features_df['patient_id'].unique()):
        mask = features_df['patient_id'] == patient_id
        df_patient = features_df[mask]
        
        # Get seizure windows for this patient
        seizure_mask = df_patient['label'] == 1
        if seizure_mask.sum() == 0:
            continue
        
        X_seizure = scaler.transform(df_patient[seizure_mask][feature_cols].values)
        X_seizure[np.isnan(X_seizure)] = 0
        X_seizure[np.isinf(X_seizure)] = 0
        y_pred = model.predict(X_seizure)
        
        # Check if any seizure window was detected
        # Group consecutive seizure windows as one "event"
        seizure_indices = np.where(seizure_mask.values)[0]
        events = []
        current_event = [seizure_indices[0]]
        
        for i in range(1, len(seizure_indices)):
            if seizure_indices[i] - seizure_indices[i-1] <= 2:  # Allow small gaps
                current_event.append(seizure_indices[i])
            else:
                events.append(current_event)
                current_event = [seizure_indices[i]]
        events.append(current_event)
        
        # Check detection for each event
        pred_idx = 0
        for event in events:
            event_length = len(event)
            event_preds = y_pred[pred_idx : pred_idx + event_length]
            detected = np.any(event_preds == 1)
            
            results["total_events"] += 1
            if detected:
                results["detected_events"] += 1
            else:
                results["missed_events"] += 1
            
            results["details"].append({
                "patient": patient_id,
                "event_windows": event_length,
                "detected": bool(detected),
                "detection_ratio": float(np.mean(event_preds == 1)),
            })
            
            pred_idx += event_length
    
    if results["total_events"] > 0:
        results["detection_rate"] = (
            results["detected_events"] / results["total_events"]
        )
    
    logger.info(f"\n  Seizure Event Detection ({model_name}):")
    logger.info(f"    Total events: {results['total_events']}")
    logger.info(f"    Detected:     {results['detected_events']}")
    logger.info(f"    Missed:       {results['missed_events']}")
    logger.info(f"    Detection rate: {results['detection_rate']:.4f}")
    
    return results
