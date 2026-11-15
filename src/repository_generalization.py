"""
Repository-Independent Generalization Experiment (Phase 2).
Evaluates whether flaky test detection models generalize to unseen software repositories
using GroupShuffleSplit with zero repository overlap between train and test.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import get_tfidf_vectorizer
from src.evaluation import evaluate_binary_classifier, plot_binary_confusion_matrix

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def parse_canonical_repo(url: str, fallback_repo: str = "") -> str:
    """Parses owner/repo from GitHub URL to guarantee unambiguous repository groups."""
    m = re.search(r"github\.com/([^/]+)/([^/]+)/", str(url))
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if isinstance(fallback_repo, str) and fallback_repo.strip() not in ["", "--", "nan"]:
        return fallback_repo.strip()
    return "unresolved_repo"


def run_repository_generalization_experiment():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "flaky_binary_dataset.csv")
    df = pd.read_csv(data_path)

    # 1. Canonical Repository ID Extraction (Ensures zero missing values)
    clean_repos = [
        parse_canonical_repo(url, fb)
        for url, fb in zip(df["url_issue"], df["repository"])
    ]
    df["canonical_repo"] = clean_repos

    print("=" * 65)
    print("PHASE 2: REPOSITORY-INDEPENDENT GENERALIZATION EXPERIMENT")
    print("=" * 65)
    print(f"Total samples: {len(df)} across {df['canonical_repo'].nunique()} unique repositories.")

    # 2. GroupShuffleSplit (80/20 group split by repository)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["canonical_repo"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_repos = set(train_df["canonical_repo"])
    test_repos = set(test_df["canonical_repo"])
    overlap = train_repos.intersection(test_repos)

    print(f"Train split: {len(train_df)} samples across {len(train_repos)} repositories (Flaky: {(train_df['label']==1).sum()}, FP: {(train_df['label']==0).sum()})")
    print(f"Test split:  {len(test_df)} samples across {len(test_repos)} repositories (Flaky: {(test_df['label']==1).sum()}, FP: {(test_df['label']==0).sum()})")
    print(f"Repository Overlap: {len(overlap)} (Zero overlap confirmed: {len(overlap) == 0})")
    assert len(overlap) == 0, "Repository leakage detected between train and test partitions!"

    # 3. Fit TF-IDF Vectorizer strictly on train split
    vectorizer = get_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train_df["combined_text"])
    X_test = vectorizer.transform(test_df["combined_text"])

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # 4. Candidate Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
                max_iter=3000
            ),
            cv=3
        )
    }

    # Load random-split baseline for direct comparison
    baseline_path = os.path.join(METRICS_DIR, "model_comparison.json")
    with open(baseline_path) as f:
        baseline_data = json.load(f)["task_a_flakiness_detection"]["models"]

    results_records = []

    for name, model in models.items():
        print(f"\nEvaluating {name} on unseen repositories...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_binary_classifier(y_test, y_pred, y_prob, name)

        # Save confusion matrix
        cm_path = os.path.join(FIGURES_DIR, f"cm_repo_indep_{name.lower().replace(' ', '_')}.png")
        plot_binary_confusion_matrix(metrics["confusion_matrix"], f"{name} (Repo-Independent)", cm_path)

        base = baseline_data[name]

        # Record comparative metrics
        results_records.append({
            "model": name,
            "evaluation_scheme": "Repository-Independent (GroupShuffleSplit)",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_repositories": len(train_repos),
            "test_repositories": len(test_repos),
            "repository_overlap": len(overlap),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "baseline_accuracy": base["accuracy"],
            "baseline_macro_f1": base["macro_f1"],
            "baseline_f1": base["f1_score"],
            "baseline_roc_auc": base["roc_auc"],
            "macro_f1_gap": round(metrics["macro_f1"] - base["macro_f1"], 4),
            "accuracy_gap": round(metrics["accuracy"] - base["accuracy"], 4)
        })

        print(f"  -> Repo-Independent | Macro-F1: {metrics['macro_f1']} (Baseline: {base['macro_f1']}, Δ: {metrics['macro_f1'] - base['macro_f1']:+.4f})")
        print(f"  -> Accuracy: {metrics['accuracy']} (Baseline: {base['accuracy']}, Δ: {metrics['accuracy'] - base['accuracy']:+.4f}) | ROC-AUC: {metrics['roc_auc']}")

    res_df = pd.DataFrame(results_records)
    csv_out = os.path.join(METRICS_DIR, "repository_generalization_results.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\n[Repository Generalization] Results saved to {csv_out}")

    # 5. Generate Visual Comparison Chart
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(models))
    w = 0.35

    base_f1 = [baseline_data[m]["macro_f1"] for m in models.keys()]
    repo_f1 = [res_df.loc[res_df["model"] == m, "macro_f1"].values[0] for m in models.keys()]

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - w / 2, base_f1, w, label="Random 80/20 Split (Optimistic / Overlapping Repos)", color="#4361EE")
    rects2 = ax.bar(x + w / 2, repo_f1, w, label="Repository-Independent Split (Strict Zero Overlap)", color="#E63946")

    ax.set_ylabel("Macro-F1 Score", weight="bold")
    ax.set_title("Generalization Gap: Random Split vs. Repository-Independent Evaluation", weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(list(models.keys()), weight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, loc="lower right")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2., h + 0.02),
                    ha="center", va="bottom", fontsize=9, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2., h + 0.02),
                    ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    comp_fig_path = os.path.join(FIGURES_DIR, "repository_generalization_comparison.png")
    plt.savefig(comp_fig_path, dpi=200)
    plt.close()
    print(f"[Repository Generalization] Comparison plot saved to {comp_fig_path}")

    return res_df


if __name__ == "__main__":
    run_repository_generalization_experiment()
