"""
Automated Tests for Dataset Loading and Schema Integrity.
"""

import os
import pandas as pd
import pytest

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_CSV = os.path.join(DATA_DIR, "raw", "raw_flaky_issues.csv")
PROCESSED_BIN = os.path.join(DATA_DIR, "processed", "flaky_binary_dataset.csv")
PROCESSED_RC = os.path.join(DATA_DIR, "processed", "root_cause_dataset.csv")


def test_raw_dataset_exists_and_non_empty():
    assert os.path.exists(RAW_CSV), f"Raw dataset not found at {RAW_CSV}"
    df = pd.read_csv(RAW_CSV)
    assert len(df) > 1000, f"Expected >1000 rows, found {len(df)}"
    assert "url_issue" in df.columns
    assert "language" in df.columns
    assert "flakiness_status" in df.columns


def test_processed_binary_dataset_labels():
    assert os.path.exists(PROCESSED_BIN)
    df = pd.read_csv(PROCESSED_BIN)
    assert len(df) > 900
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["flakiness_status"].unique()) == {"T", "F"}
    assert "ND" not in df["flakiness_status"].values
    assert df["combined_text"].isna().sum() == 0


def test_processed_root_cause_dataset_classes():
    assert os.path.exists(PROCESSED_RC)
    df = pd.read_csv(PROCESSED_RC)
    assert len(df) > 500
    assert df["root_cause_category"].nunique() >= 10
    assert "--" not in df["root_cause_category"].values
    assert "Hard to Classify" not in df["root_cause_category"].values
