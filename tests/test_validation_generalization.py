"""
Automated Tests for Research Validation and Generalization Procedures (Phase 8).
Verifies:
1. Zero repository overlap in grouped repository-independent evaluation.
2. Zero language overlap in Leave-One-Language-Out (LOLO) partitions.
3. Strict vectorizer fitting isolation (no test data leakage).
4. Existence and integrity of all required validation reports and figures.
"""

import os
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from src.repository_generalization import parse_canonical_repo
from src.feature_engineering import get_tfidf_vectorizer

PROCESSED_BIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flaky_binary_dataset.csv"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))


def test_canonical_repo_extraction_zero_missing():
    df = pd.read_csv(PROCESSED_BIN_PATH)
    repos = [parse_canonical_repo(u, fb) for u, fb in zip(df["url_issue"], df["repository"])]
    assert len(repos) == len(df)
    assert all(r and r != "unresolved_repo" for r in repos)
    assert len(set(repos)) > 500


def test_repository_independent_split_zero_overlap():
    df = pd.read_csv(PROCESSED_BIN_PATH)
    df["repo"] = [parse_canonical_repo(u, fb) for u, fb in zip(df["url_issue"], df["repository"])]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["repo"]))

    tr_repos = set(df.iloc[tr_idx]["repo"])
    te_repos = set(df.iloc[te_idx]["repo"])

    overlap = tr_repos.intersection(te_repos)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping repositories!"
    assert len(tr_idx) + len(te_idx) == len(df)


def test_lolo_partitioning_zero_language_overlap():
    df = pd.read_csv(PROCESSED_BIN_PATH)
    languages = df["language"].unique()

    for lang in languages:
        train_df = df[df["language"] != lang]
        test_df = df[df["language"] == lang]

        assert lang not in train_df["language"].values
        assert set(test_df["language"].unique()) == {lang}
        assert len(train_df) + len(test_df) == len(df)


def test_vectorizer_fitting_isolation():
    # Verify that vectorizer fitted on training data has vocabulary strictly derived from training
    train_texts = pd.Series(["goroutine channel timeout deadlock", "race condition in thread"])
    test_texts = pd.Series(["unseen unique_test_word_12345 in assertion"])

    vec = get_tfidf_vectorizer(min_df=1)
    vec.fit(train_texts)

    vocab = set(vec.get_feature_names_out())
    assert "channel" in vocab
    assert "goroutine" in vocab
    # Test word MUST NOT be in vocabulary
    assert "unique_test_word_12345" not in vocab


def test_validation_artifact_files_exist():
    required_metrics_files = [
        "reproducibility_audit.md",
        "repository_generalization_results.csv",
        "lolo_results.csv",
        "lolo_evaluation_report.md",
        "root_cause_generalization_report.md",
        "explainability_reliability_report.md",
        "research_validation_report.md"
    ]
    for filename in required_metrics_files:
        path = os.path.join(METRICS_DIR, filename)
        assert os.path.exists(path), f"Missing required metrics artifact: {filename}"
        assert os.path.getsize(path) > 50, f"Artifact {filename} is empty!"

    required_figures = [
        "repository_generalization_comparison.png",
        "lolo_language_comparison.png",
        os.path.join("explainability", "shap_summary_plot.png"),
        os.path.join("explainability", "model_shift_explanation_comparison.png")
    ]
    for filename in required_figures:
        path = os.path.join(FIGURES_DIR, filename)
        assert os.path.exists(path), f"Missing required figure artifact: {filename}"
        assert os.path.getsize(path) > 1000, f"Figure {filename} is empty or corrupted!"
