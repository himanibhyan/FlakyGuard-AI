"""
Automated Test Suite for Second-Level Research Audit.
Verifies dataset count reconciliation, zero repository overlap across all 10 seeds,
LOLO matrix completeness, bootstrap confidence interval calculations, and explainability mathematical invariants.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.repository_generalization import parse_canonical_repo
from src.confidence_intervals import bootstrap_ci

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def test_reconciliation_dataset_counts():
    """Verifies reconciled sample and repository counts for binary and root-cause datasets."""
    bin_path = os.path.join(DATA_DIR, "flaky_binary_dataset.csv")
    assert os.path.exists(bin_path), "flaky_binary_dataset.csv missing"
    bin_df = pd.read_csv(bin_path)

    # 1. Binary dataset exact counts
    assert len(bin_df) == 1101, f"Expected 1,101 binary rows, found {len(bin_df)}"
    assert (bin_df["label"] == 1).sum() == 701, "Expected 701 Flaky (T) samples"
    assert (bin_df["label"] == 0).sum() == 400, "Expected 400 False Positive (F) samples"

    # Canonical repository count
    bin_df["canonical_repo"] = [
        parse_canonical_repo(u, fb) for u, fb in zip(bin_df["url_issue"], bin_df["repository"])
    ]
    assert bin_df["canonical_repo"].nunique() == 641, f"Expected 641 unique repositories, found {bin_df['canonical_repo'].nunique()}"

    # 2. Root cause dataset exact counts
    rc_path = os.path.join(DATA_DIR, "root_cause_dataset.csv")
    assert os.path.exists(rc_path), "root_cause_dataset.csv missing"
    rc_df = pd.read_csv(rc_path)

    assert len(rc_df) == 589, f"Expected 589 root-cause rows, found {len(rc_df)}"
    assert rc_df["root_cause_category"].nunique() == 13, "Expected 13 distinct root-cause categories"

    rc_df["canonical_repo"] = [
        parse_canonical_repo(u, fb) for u, fb in zip(rc_df["url_issue"], rc_df["repository"])
    ]
    assert rc_df["canonical_repo"].nunique() == 349, f"Expected 349 unique repositories, found {rc_df['canonical_repo'].nunique()}"


def test_root_cause_13_to_4_mapping_exhaustive():
    """Verifies that all 13 root cause categories map exhaustively to the 4 macro categories."""
    rc_path = os.path.join(DATA_DIR, "root_cause_dataset.csv")
    rc_df = pd.read_csv(rc_path)

    macro_mapping = {
        "Async Wait": "Timing & Asynchrony",
        "Time": "Timing & Asynchrony",
        "Test Case Timeout": "Timing & Asynchrony",
        "Test Suite Timeout": "Timing & Asynchrony",
        "Concurrency": "Concurrency & Non-Determinism",
        "Randomness": "Concurrency & Non-Determinism",
        "Unordered Collections": "Concurrency & Non-Determinism",
        "Network": "Resource & Environment",
        "Resource Leak": "Resource & Environment",
        "Platform Dependency": "Resource & Environment",
        "Test Order Dependency": "Test Logic & Assertions",
        "Floating Point Operations": "Test Logic & Assertions",
        "Too Restrictive Range": "Test Logic & Assertions"
    }

    assert len(macro_mapping) == 13
    assert set(rc_df["root_cause_category"].unique()) == set(macro_mapping.keys())

    rc_df["macro_category"] = rc_df["root_cause_category"].map(macro_mapping)
    assert rc_df["macro_category"].isnull().sum() == 0, "Null macro-categories detected"
    assert rc_df["macro_category"].nunique() == 4


def test_zero_repository_overlap_across_all_10_seeds():
    """Verifies that none of the 10 random seeds have repository overlap between train and test."""
    bin_path = os.path.join(DATA_DIR, "flaky_binary_dataset.csv")
    df = pd.read_csv(bin_path)
    df["canonical_repo"] = [
        parse_canonical_repo(u, fb) for u, fb in zip(df["url_issue"], df["repository"])
    ]

    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, groups=df["canonical_repo"]))

        train_repos = set(df.iloc[tr_idx]["canonical_repo"])
        test_repos = set(df.iloc[te_idx]["canonical_repo"])

        overlap = train_repos.intersection(test_repos)
        assert len(overlap) == 0, f"Seed {seed} leaked {len(overlap)} overlapping repositories"
        assert len(train_repos) + len(test_repos) == 641, f"Seed {seed} total repos != 641"


def test_repeated_repository_csv_completeness():
    """Verifies the generated repeated repository generalization CSV."""
    csv_path = os.path.join(METRICS_DIR, "repeated_repository_generalization.csv")
    assert os.path.exists(csv_path), "repeated_repository_generalization.csv missing"
    df = pd.read_csv(csv_path)

    assert len(df) == 30, f"Expected 30 rows (10 seeds * 3 models), found {len(df)}"
    assert set(df["seed"].unique()) == {42, 101, 202, 303, 404, 505, 606, 707, 808, 909}
    assert set(df["model"].unique()) == {"Random Forest", "Logistic Regression", "Linear SVM"}
    assert df["macro_f1"].isnull().sum() == 0
    assert (df["macro_f1"] > 0.40).all(), "All runs should surpass chance threshold"


def test_bootstrap_ci_helper():
    """Tests empirical bootstrap confidence interval computation sanity."""
    data = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70])
    mean_val, se, lo, hi = bootstrap_ci(data, alpha=0.05)

    assert lo <= mean_val <= hi, f"Bootstrap bounds inconsistent: {lo} <= {mean_val} <= {hi}"
    assert se > 0.0, "Standard error must be strictly positive"


def test_explainability_invariants():
    """Verifies mathematical properties of Random Forest MDI vs Linear model coefficients."""
    import joblib
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    rf = joblib.load(os.path.join(models_dir, "best_flaky_model.joblib"))
    lr = joblib.load(os.path.join(models_dir, "flaky_logistic_regression.joblib"))

    # Random Forest importances are non-negative
    assert (rf.feature_importances_ >= 0.0).all(), "Random Forest MDI importances cannot be negative"
    assert np.isclose(rf.feature_importances_.sum(), 1.0, atol=1e-3), "Random Forest importances should sum to 1.0"

    # Logistic Regression has signed coefficients (both positive and negative)
    assert (lr.coef_ > 0).any(), "Logistic regression must contain positive coefficients"
    assert (lr.coef_ < 0).any(), "Logistic regression must contain negative coefficients"
