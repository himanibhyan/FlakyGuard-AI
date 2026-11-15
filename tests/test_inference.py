"""
Automated Tests for Inference Engine and Explainability.
"""

import os
import pytest
from src.inference import FlakyGuardEngine


@pytest.fixture(scope="module")
def engine():
    return FlakyGuardEngine.get_instance()


def test_predict_flakiness_positive(engine):
    res = engine.predict_flakiness(
        title="TestRaceCondition is flaky on CI",
        description="Goroutine deadlock occurs when channel buffer is full during stress tests.",
        language="Go"
    )
    assert "prediction" in res
    assert res["prediction"] in ["Flaky", "False Positive"]
    assert 0.0 <= res["flaky_probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["influential_words"]) > 0


def test_predict_flakiness_empty_inputs(engine):
    # Tests graceful handling of empty or blank input
    res = engine.predict_flakiness(title="", description="", language="")
    assert "prediction" in res
    assert res["prediction"] in ["Flaky", "False Positive"]
    assert res["confidence"] >= 0.5


def test_predict_root_cause(engine):
    res = engine.predict_root_cause(
        title="Intermittent socket timeout during test suite execution",
        description="Async wait deadline exceeded after 10000ms waiting for server response",
        language="JavaScript"
    )
    assert "predicted_root_cause" in res
    assert "predicted_macro_category" in res
    assert isinstance(res["predicted_root_cause"], str)
    assert isinstance(res["predicted_macro_category"], str)
    assert res["predicted_macro_category"] in [
        "Timing & Asynchrony",
        "Concurrency & Non-Determinism",
        "Resource & Environment",
        "Test Logic & Assertions",
        "Other / Unknown"
    ]
    assert len(res["probability_distribution"]) >= 10
    total_prob = sum(res["probability_distribution"].values())
    assert 0.98 <= total_prob <= 1.02


def test_predict_flakiness_all_model_variants(engine):
    """Verifies inference across all supported binary model variants."""
    models = ["Random Forest", "Logistic Regression", "Linear SVM"]
    for m in models:
        res = engine.predict_flakiness(
            title="Deadlock in channel queue",
            description="Concurrent write operation fails intermittently",
            language="Go",
            model_name=m
        )
        assert res["prediction"] in ["Flaky", "False Positive"]
        assert res["model_used"] == m
        assert res["confidence"] >= 0.5


def test_inference_cli_execution():
    """Verifies that the command-line entrypoint executes successfully."""
    import subprocess
    import json
    import sys

    cmd = [
        sys.executable,
        "src/inference.py",
        "--title", "TestFlakyNetwork failure",
        "--description", "Connection reset by peer after 5000ms",
        "--language", "Python",
        "--root-cause",
        "--json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".", MPLCONFIGDIR="/tmp/mpl"))
    assert result.returncode == 0, f"CLI exited with error: {result.stderr}"
    data = json.loads(result.stdout)
    assert "binary_detection" in data
    assert "root_cause_classification" in data
    assert data["binary_detection"]["prediction"] in ["Flaky", "False Positive"]
    assert data["root_cause_classification"]["predicted_root_cause"] != ""

