"""
Visualization Module.

Generates publication-quality figures for the EEG seizure detection pipeline:
1. Confusion matrices (per model)
2. ROC curves (all models overlay)
3. Precision-Recall curves (all models overlay)
4. Feature importance analysis
5. Per-patient performance heatmap
6. Class distribution plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import logging

logger = logging.getLogger("EEG-Pipeline")

# Global style
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

COLORS = {
    "Logistic Regression": "#3498db",
    "Linear SVM": "#e74c3c",
    "RBF SVM": "#2ecc71",
}


def plot_confusion_matrices(all_metrics, figures_dir):
    """Plot confusion matrices for all models side by side."""
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, metrics) in zip(axes, all_metrics.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Non-Seizure", "Seizure"],
            yticklabels=["Non-Seizure", "Seizure"],
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"{model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    
    plt.suptitle("Confusion Matrices - Test Set", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_roc_curves(all_metrics, figures_dir):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for model_name, metrics in all_metrics.items():
        if "roc_curve" in metrics:
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            auc = metrics["roc_auc"]
            color = COLORS.get(model_name, "#333333")
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", 
                    color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves - Test Set", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_pr_curves(all_metrics, figures_dir):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for model_name, metrics in all_metrics.items():
        if "pr_curve" in metrics:
            prec = metrics["pr_curve"]["precision"]
            rec = metrics["pr_curve"]["recall"]
            auc = metrics["pr_auc"]
            color = COLORS.get(model_name, "#333333")
            ax.plot(rec, prec, label=f"{model_name} (AP={auc:.3f})",
                    color=color, linewidth=2)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Test Set", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(figures_dir, "pr_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_feature_importance(model, feature_names, figures_dir, 
                            model_name="Linear SVM", top_n=25):
    """
    Plot feature importance based on SVM coefficients or permutation importance.
    
    For linear models, uses the absolute coefficient values.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        if hasattr(model, "coef_"):
            # Linear models — use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            logger.info(f"  {model_name}: No coef_ attribute, skipping feature importance.")
            plt.close()
            return
    except Exception as e:
        logger.warning(f"  Could not extract feature importance for {model_name}: {e}")
        plt.close()
        return
    
    # Get top N features
    top_indices = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in top_indices]
    top_values = importances[top_indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_values)))
    ax.barh(range(len(top_values)), top_values, color=colors)
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Absolute Coefficient Value")
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}",
        fontsize=14, fontweight="bold"
    )
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(figures_dir, f"feature_importance_{safe_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_per_patient_results(patient_results_df, figures_dir, model_name="Best Model"):
    """Plot per-patient F1 scores as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = patient_results_df.sort_values("patient_id")
    
    colors = []
    for _, row in df.iterrows():
        if row["n_seizure"] == 0:
            colors.append("#95a5a6")  # Gray for no seizures
        elif row["f1"] >= 0.7:
            colors.append("#2ecc71")  # Green for good
        elif row["f1"] >= 0.4:
            colors.append("#f39c12")  # Orange for moderate
        else:
            colors.append("#e74c3c")  # Red for poor
    
    bars = ax.bar(range(len(df)), df["f1"], color=colors, edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["patient_id"], rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Per-Patient F1 Scores — {model_name}",
        fontsize=14, fontweight="bold"
    )
    ax.axhline(y=df["f1"].mean(), color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean F1: {df['f1'].mean():.3f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add count annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["f1"] + 0.02, f"S:{row['n_seizure']}", 
                ha="center", fontsize=7, color="#555555")
    
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(figures_dir, f"per_patient_f1_{safe_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_class_distribution(features_df, figures_dir):
    """Plot class distribution across the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    counts = features_df['label'].value_counts()
    labels = ["Non-Seizure", "Seizure"]
    colors = ["#3498db", "#e74c3c"]
    
    axes[0].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, explode=(0, 0.05))
    axes[0].set_title("Overall Class Distribution", fontsize=14, fontweight="bold")
    
    # Per-patient distribution
    patient_counts = features_df.groupby("patient_id")["label"].value_counts().unstack(fill_value=0)
    if 0 in patient_counts.columns and 1 in patient_counts.columns:
        patient_counts.plot(
            kind="bar", stacked=True, ax=axes[1],
            color=colors, edgecolor="white", linewidth=0.5
        )
    elif 0 in patient_counts.columns:
        patient_counts[0].plot(
            kind="bar", ax=axes[1],
            color=colors[0], edgecolor="white", linewidth=0.5
        )
    
    axes[1].set_title("Per-Patient Window Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Patient")
    axes[1].set_ylabel("Number of Windows")
    axes[1].legend(["Non-Seizure", "Seizure"], loc="upper right")
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_metrics_comparison(all_metrics, figures_dir):
    """Plot bar chart comparing all models across key metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC"]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        color = COLORS.get(model_name, "#333333")
        offset = (i - len(all_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=color,
                      edgecolor="white", linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def generate_all_figures(all_metrics, features_df, models, feature_names, 
                         patient_results, figures_dir):
    """Generate all visualization figures."""
    os.makedirs(figures_dir, exist_ok=True)
    
    logger.info("\nGenerating figures...")
    
    # 1. Confusion matrices
    plot_confusion_matrices(all_metrics, figures_dir)
    
    # 2. ROC curves
    plot_roc_curves(all_metrics, figures_dir)
    
    # 3. PR curves
    plot_pr_curves(all_metrics, figures_dir)
    
    # 4. Model comparison
    plot_metrics_comparison(all_metrics, figures_dir)
    
    # 5. Class distribution
    plot_class_distribution(features_df, figures_dir)
    
    # 6. Feature importance (for linear models)
    for model_name, model in models.items():
        plot_feature_importance(model, feature_names, figures_dir, model_name)
    
    # 7. Per-patient results
    for model_name, df in patient_results.items():
        plot_per_patient_results(df, figures_dir, model_name)
    
    logger.info("All figures generated.")
