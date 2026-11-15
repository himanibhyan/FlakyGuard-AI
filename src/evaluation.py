"""
Evaluation Module for FlakyGuard AI.
Calculates classification metrics, generates ROC/PR curves, confusion matrices,
and saves publication-ready artifact plots and JSON reports.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    precision_recall_curve
)

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str
) -> Dict[str, Any]:
    """
    Computes full evaluation suite for Task A (Binary Flakiness Detection):
    Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, PR-AUC, Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=["False Positive", "Flaky"],
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "roc_auc": float(round(roc_auc, 4)) if roc_auc is not None else "N/A",
        "pr_auc": float(round(pr_auc, 4)) if pr_auc is not None else "N/A",
        "confusion_matrix": cm,
        "classification_report": clf_report
    }
    return metrics


def evaluate_multiclass_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str
) -> Dict[str, Any]:
    """
    Computes evaluation metrics for Task B (Root-Cause Classification):
    Accuracy, Macro-F1, Weighted-F1, Per-class Precision, Recall, and F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "confusion_matrix": cm,
        "per_class": {
            cls: {
                "precision": float(round(clf_report[cls]["precision"], 4)),
                "recall": float(round(clf_report[cls]["recall"], 4)),
                "f1-score": float(round(clf_report[cls]["f1-score"], 4)),
                "support": int(clf_report[cls]["support"])
            }
            for cls in class_names if cls in clf_report
        },
        "classification_report": clf_report
    }
    return metrics


def plot_binary_confusion_matrix(
    cm: List[List[int]],
    model_name: str,
    save_path: str
) -> None:
    """Plots and saves confusion matrix heatmap for binary classification."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["False Positive (0)", "Flaky (1)"],
        yticklabels=["False Positive (0)", "Flaky (1)"]
    )
    plt.title(f"Confusion Matrix: {model_name}", weight="bold", pad=10)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multiclass_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    model_name: str,
    save_path: str
) -> None:
    """Plots and saves confusion matrix heatmap for root-cause classification."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Root-Cause Confusion Matrix: {model_name}", weight="bold", pad=12)
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_pr_curves(
    curves_data: Dict[str, Dict[str, np.ndarray]],
    save_path_roc: str,
    save_path_pr: str
) -> None:
    """Plots comparative ROC and Precision-Recall curves across models."""
    # ROC Curves
    plt.figure(figsize=(7, 5))
    for name, data in curves_data.items():
        if "y_prob" in data and data["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
            auc = roc_auc_score(data["y_true"], data["y_prob"])
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", weight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path_roc), exist_ok=True)
    plt.savefig(save_path_roc, dpi=200)
    plt.close()

    # PR Curves
    plt.figure(figsize=(7, 5))
    for name, data in curves_data.items():
        if "y_prob" in data and data["y_prob"] is not None:
            prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
            ap = average_precision_score(data["y_true"], data["y_prob"])
            plt.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.3f})")
    plt.title("Precision-Recall (PR) Curve Comparison", weight="bold", pad=12)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path_pr), exist_ok=True)
    plt.savefig(save_path_pr, dpi=200)
    plt.close()
