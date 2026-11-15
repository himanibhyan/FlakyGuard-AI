"""
Automated Tests for Data Preprocessing and Text Normalization.
"""

import pandas as pd
import pytest
from src.data_preprocessing import clean_text, parse_url_context, prepare_binary_flakiness_dataset


def test_clean_text_removes_urls_and_markdown():
    raw_sample = "Test failed at https://ci.example.com/build/123 with `assertion error` ## Header"
    cleaned = clean_text(raw_sample)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "`" not in cleaned
    assert "#" not in cleaned
    assert "test failed at" in cleaned


def test_clean_text_handles_missing_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("--") == ""
    assert clean_text("nan") == ""


def test_parse_url_context():
    url = "https://github.com/DataDog/dd-trace-py/pull/193"
    ctx = parse_url_context(url)
    assert "dd trace py" in ctx
    assert "pull" in ctx
    assert "193" in ctx


def test_binary_preprocessing_no_leakage():
    mock_df = pd.DataFrame([
        {
            "issue_id": 1,
            "repository": "repo1",
            "language": "python",
            "flakiness_status": "T",
            "url_issue": "https://github.com/org/repo1/issues/10",
            "issue_title": "Flaky timeout in worker pool",
            "issue_body": "Context deadline exceeded after 5 seconds",
            "problem_description_excerpt": "Annotator secret text",
            "root_cause_category": "Async Wait"
        },
        {
            "issue_id": 2,
            "repository": "repo2",
            "language": "go",
            "flakiness_status": "F",
            "url_issue": "https://github.com/org/repo2/issues/20",
            "issue_title": "Button alignment bug",
            "issue_body": "CSS margin is too wide on mobile devices",
            "problem_description_excerpt": "--",
            "root_cause_category": "--"
        }
    ])

    binary_out = prepare_binary_flakiness_dataset(mock_df)
    assert len(binary_out) == 2
    # Ensure annotator excerpt is NOT in model feature columns
    assert "problem_description_excerpt" not in binary_out.columns
    assert "root_cause_category" not in binary_out.columns
    assert binary_out.loc[binary_out["issue_id"] == 1, "label"].values[0] == 1
    assert binary_out.loc[binary_out["issue_id"] == 2, "label"].values[0] == 0
