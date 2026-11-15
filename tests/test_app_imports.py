"""
Automated Tests for Streamlit Page Imports and App Components.
"""

import importlib
import pytest


def test_app_and_page_imports():
    modules = [
        "src.data_acquisition",
        "src.data_preprocessing",
        "src.feature_engineering",
        "src.flaky_detection",
        "src.root_cause_classifier",
        "src.model_training",
        "src.evaluation",
        "src.explainability",
        "src.inference",
        "src.ui_theme"
    ]
    for mod in modules:
        m = importlib.import_module(mod)
        assert m is not None, f"Failed to import {mod}"
