"""
Automated Tests for Model Persistence and Performance Integrity.
"""

import os
import joblib
import numpy as np
import pytest

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def test_saved_model_artifacts_exist():
    expected_artifacts = [
        "flaky_vectorizer.joblib",
        "best_flaky_model.joblib",
        "flaky_logistic_regression.joblib",
        "flaky_linear_svm.joblib",
        "flaky_random_forest.joblib",
        "rc_vectorizer.joblib",
        "rc_label_encoder.joblib",
        "best_rc_model.joblib",
        "rc_logistic_regression.joblib",
        "rc_linear_svm.joblib"
    ]
    for artifact in expected_artifacts:
        path = os.path.join(MODELS_DIR, artifact)
        assert os.path.exists(path), f"Missing artifact: {artifact}"
        assert os.path.getsize(path) > 100, f"Artifact {artifact} is empty or corrupted"


def test_vectorizers_load_and_transform():
    vec_a = joblib.load(os.path.join(MODELS_DIR, "flaky_vectorizer.joblib"))
    sample_text = ["test flaky wait timeout in goroutine channel"]
    X = vec_a.transform(sample_text)
    assert X.shape[0] == 1
    assert X.shape[1] > 1000  # vocabulary size
    assert X.nnz > 0


def test_model_prediction_outputs():
    vec_a = joblib.load(os.path.join(MODELS_DIR, "flaky_vectorizer.joblib"))
    model_a = joblib.load(os.path.join(MODELS_DIR, "best_flaky_model.joblib"))
    X = vec_a.transform(["intermittent race condition in mutex lock"])
    pred = model_a.predict(X)
    assert pred[0] in [0, 1]
    if hasattr(model_a, "predict_proba"):
        probs = model_a.predict_proba(X)[0]
        assert len(probs) == 2
        assert np.isclose(np.sum(probs), 1.0, atol=1e-3)
