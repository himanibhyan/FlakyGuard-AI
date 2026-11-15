"""
Model Training and Evaluation Pipeline for FlakyGuard AI.
Orchestrates training for Task A (Flakiness Detection) and Task B (Root Cause Classification),
evaluates metrics, plots curves/confusion matrices, and serializes trained models.
"""

import os
import json
import pandas as pd
import numpy as np

from src.flaky_detection import FlakyDetector
from src.root_cause_classifier import RootCauseClassifier
from src.evaluation import (
    plot_binary_confusion_matrix,
    plot_multiclass_confusion_matrix,
    plot_roc_pr_curves
)
from src.explainability import FlakyExplainer

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def run_training_pipeline() -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("=" * 60)
    print("FLAKYGUARD AI: MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. TASK A: FLAKINESS DETECTION (T vs F)
    # ---------------------------------------------------------
    binary_path = os.path.join(PROCESSED_DIR, "flaky_binary_dataset.csv")
    print(f"\n[Task A] Loading binary dataset from {binary_path}...")
    binary_df = pd.read_csv(binary_path)

    detector = FlakyDetector(random_state=42)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = detector.prepare_data(binary_df)
    print(f"[Task A] Partitioned into Train ({X_train_bin.shape[0]} samples) and Test ({X_test_bin.shape[0]} samples)")

    bin_results = detector.train_and_evaluate_all(X_train_bin, X_test_bin, y_train_bin, y_test_bin)
    detector.save_artifacts(MODELS_DIR)

    # Plot Confusion Matrices for Task A models
    for name, res in bin_results.items():
        cm_path = os.path.join(FIGURES_DIR, f"cm_task_a_{name.lower().replace(' ', '_')}.png")
        plot_binary_confusion_matrix(res["confusion_matrix"], name, cm_path)

    # Plot ROC & PR Curves for Task A
    roc_path = os.path.join(FIGURES_DIR, "task_a_roc_curves.png")
    pr_path = os.path.join(FIGURES_DIR, "task_a_pr_curves.png")
    plot_roc_pr_curves(detector.curves_data, roc_path, pr_path)

    # ---------------------------------------------------------
    # 2. TASK B: ROOT-CAUSE CLASSIFICATION
    # ---------------------------------------------------------
    rc_path = os.path.join(PROCESSED_DIR, "root_cause_dataset.csv")
    print(f"\n[Task B] Loading root cause dataset from {rc_path}...")
    rc_df = pd.read_csv(rc_path)

    rc_classifier = RootCauseClassifier(random_state=42)
    X_train_rc, X_test_rc, y_train_rc, y_test_rc = rc_classifier.prepare_data(rc_df)
    print(f"[Task B] Partitioned into Train ({X_train_rc.shape[0]} samples) and Test ({X_test_rc.shape[0]} samples)")

    rc_results = rc_classifier.train_and_evaluate_all(X_train_rc, X_test_rc, y_train_rc, y_test_rc)
    rc_classifier.save_artifacts(MODELS_DIR)

    # Plot Root Cause Confusion Matrices
    for name, res in rc_results.items():
        cm_path = os.path.join(FIGURES_DIR, f"cm_task_b_{name.lower().replace(' ', '_')}.png")
        plot_multiclass_confusion_matrix(res["confusion_matrix"], rc_classifier.class_names, name, cm_path)

    # ---------------------------------------------------------
    # 3. GLOBAL EXPLAINABILITY ARTIFACTS
    # ---------------------------------------------------------
    print("\n[Explainability] Generating global interpretability plots...")
    explainer_path = os.path.join(FIGURES_DIR, "global_feature_importance_flakiness.png")
    FlakyExplainer.plot_global_feature_importance(
        detector.vectorizer,
        detector.best_model,
        f"Task A Best Model ({detector.best_model_name})",
        explainer_path,
        top_n=15
    )

    # ---------------------------------------------------------
    # 4. SAVE UNIFIED BENCHMARK REPORT
    # ---------------------------------------------------------
    benchmark_report = {
        "task_a_flakiness_detection": {
            "best_model": detector.best_model_name,
            "models": bin_results
        },
        "task_b_root_cause_classification": {
            "best_model": rc_classifier.best_model_name,
            "classes": rc_classifier.class_names,
            "models": rc_results
        }
    }

    report_path = os.path.join(METRICS_DIR, "model_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\n[Training] Unified benchmark report saved to: {report_path}")

    return benchmark_report


if __name__ == "__main__":
    run_training_pipeline()
