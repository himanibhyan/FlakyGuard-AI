"""
FLAKYGUARD AI: Explainable Flaky Test Detection & Root-Cause Classification
Main Application Entrypoint.
Based on Barbosa et al., IEEE Transactions on Software Engineering (TSE), 2023.
"""

import os
import json
import streamlit as st
import pandas as pd
from src.ui_theme import apply_theme, render_metric_card

st.set_page_config(
    page_title="FlakyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# Load summary metrics
METRICS_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "metrics", "model_comparison.json")
PREPROC_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "metrics", "preprocessing_summary.json")

preproc_data = {}
model_data = {}
if os.path.exists(PREPROC_PATH):
    with open(PREPROC_PATH) as f:
        preproc_data = json.load(f)
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        model_data = json.load(f)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=48)
    st.title("FlakyGuard AI")
    st.caption("Explainable Flaky Test Intelligence")
    st.markdown("---")
    st.markdown("**Paper Reference:**")
    st.markdown(
        "*Test Flakiness Across Programming Languages*\n"
        "Barbosa et al., **IEEE TSE 2023**"
    )
    st.markdown("---")
    st.markdown("**Status:** Local Classical ML Active")
    st.markdown(f"**Best Flakiness Model:** `{model_data.get('task_a_flakiness_detection', {}).get('best_model', 'Random Forest')}`")
    st.markdown(f"**Best Root-Cause Model:** `{model_data.get('task_b_root_cause_classification', {}).get('best_model', 'Logistic Regression')}`")

# Header
st.title("🛡️ FLAKYGUARD AI")
st.subheader("Machine Learning for Flaky Test Detection & Explainable Root-Cause Classification")
st.markdown(
    """
    > **Research Context:** This system analyzes software engineering **GitHub issue reports** related to test flakiness,
    > categorizing genuine non-deterministic test failures from false positives, predicting probable root causes,
    > and explaining linguistic triggers via interpretable machine learning.
    """
)

st.markdown("---")

# Key KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Total Dataset Issues", "1,513", "Deduplicated TSE22 issues")
with c2:
    render_metric_card("Programming Languages", "5", "C, Go, Java, JS, Python")
with c3:
    render_metric_card("Confirmed Flaky (T)", "701", "Ground truth verified")
with c4:
    render_metric_card("Root-Cause Categories", "13", "Distinct failure modes")
with c5:
    best_f1 = model_data.get("task_a_flakiness_detection", {}).get("models", {}).get("Random Forest", {}).get("macro_f1", 0.6848)
    render_metric_card("Best Task A Macro-F1", f"{best_f1:.3f}", "Random Forest Classifier")

st.markdown("---")

# Feature Highlights & Navigation Guide
st.markdown("### 🧭 System Navigation & Capabilities")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        #### 📊 [1. Overview Dashboard](Overview)
        - Global flakiness distributions across programming languages
        - Frequency of all 13 confirmed root-cause categories
        - Benchmark comparative matrix for classical ML architectures
        - Empirical dataset insights & imbalance characteristics

        #### 🔍 [2. Flaky Test Detector](Flaky_Test_Detector)
        - Interactive real-time test issue diagnosis
        - Calibrated flakiness probability vs raw confidence score
        - Multi-class root-cause probability distribution
        - Token-level influence explanations for predictions
        """
    )

with col_b:
    st.markdown(
        """
        #### 📈 [3. Model Evaluation](Model_Evaluation)
        - Complete performance comparison: Accuracy, Precision, Recall, F1, Macro-F1
        - ROC and Precision-Recall (PR) curves
        - Confusion matrices for binary and multi-class models
        - Per-category breakdown for rare root causes

        #### 🔬 [4. Explainability](Explainability)
        - Global feature importance (top positive & negative vocabulary weights)
        - Local LIME instance explanations
        - Interactive keyword impact inspector

        #### 🗃️ [5. Dataset Explorer](Dataset_Explorer)
        - Filter by language, label (T / F / ND), and root cause
        - Full-text search across issue descriptions
        """
    )

st.markdown("---")
st.markdown(
    "<small style='color: #64748B;'>FlakyGuard AI • Academic Machine Learning Implementation • IEEE TSE 2023</small>",
    unsafe_allow_html=True
)
