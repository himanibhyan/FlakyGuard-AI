"""
Page 4: Explainability & Interpretability for FlakyGuard AI.
Visualizes global vocabulary coefficients, SHAP summary attributions,
interactive local LIME sample explanations, and attribution stability under distribution shift.
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.ui_theme import apply_theme
from src.inference import FlakyGuardEngine
from src.explainability import FlakyExplainer

st.set_page_config(page_title="Explainability | FlakyGuard AI", page_icon="🔬", layout="wide")
apply_theme()

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures")
XAI_FIGURES_DIR = os.path.join(FIGURES_DIR, "explainability")

@st.cache_resource
def get_engine():
    return FlakyGuardEngine.get_instance()

engine = get_engine()

st.title("🔬 Explainable AI (XAI) & Interpretability")
st.markdown(
    "Unpack the internal decision mechanisms of the trained models. Inspect **global feature importances** "
    "(via SHAP & linear weights), **local instance explanations** (via LIME), and **attribution stability under distribution shift**."
)

st.markdown("---")

tab_global, tab_local, tab_shift = st.tabs([
    "🌐 Global Importances (SHAP & Weights)",
    "🎯 Local Explanations (LIME)",
    "⚖️ Attribution Stability Under Shift"
])

with tab_global:
    st.subheader("Global Feature Importance & Vocabulary Polarities")

    col_w, col_shap = st.columns(2)
    with col_w:
        global_fig_path = os.path.join(FIGURES_DIR, "global_feature_importance_flakiness.png")
        if os.path.exists(global_fig_path):
            st.image(global_fig_path, caption="Top Model Feature Weights (Logistic Regression)", use_container_width=True)

    with col_shap:
        shap_fig_path = os.path.join(XAI_FIGURES_DIR, "shap_summary_plot.png")
        if os.path.exists(shap_fig_path):
            st.image(shap_fig_path, caption="SHAP Global Impact (Mean |SHAP Value| Across Test Issues)", use_container_width=True)

    st.markdown("---")
    st.markdown("##### Discriminative Vocabulary Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Core Flaky Predictors (Positive Attribution):**")
        st.markdown(
            "- Tokens like `flaky`, `timeout`, `race`, `goroutine`, `wait`, `sleep`, `intermittent`, `concurrent` "
            "consistently act as strong indicators of genuine test flakiness.\n"
            "- These terms correlate with asynchronous wait states, resource contention, and thread scheduling non-determinism."
        )
    with col2:
        st.markdown("**Core False Positive Predictors (Negative Attribution):**")
        st.markdown(
            "- Terms like `fix`, `add`, `update`, `support`, `modal`, `crash`, `syntax`, `null` "
            "consistently shift probability towards False Positive.\n"
            "- These terms correspond to deterministic functional defects, PR refactorings, or standard feature updates."
        )

with tab_local:
    st.subheader("Interactive Local LIME Instance Explainer")
    st.markdown(
        "Enter any custom issue text or choose a test report to compute a real-time LIME explanation. "
        "LIME generates perturbed variations around the instance to estimate local linear feature attribution."
    )

    sample_texts = [
        "TestAsyncChannelTimeout is intermittently failing due to goroutine race condition and context deadline exceeded.",
        "Fix UI button alignment and prevent TypeError when form is submitted without required inputs.",
        "TestWorkerPool failed once on Travis CI runner because port 8080 was already occupied by previous test."
    ]
    selected_sample = st.selectbox("Select Benchmark Example or Type Below:", sample_texts)
    custom_text = st.text_area("Target Issue Report Text for LIME Analysis:", value=selected_sample, height=100)

    if st.button("Generate LIME Attribution", type="primary"):
        with st.spinner("Computing LIME local perturbations..."):
            explainer = FlakyExplainer(class_names=["False Positive", "Flaky"])

            def predict_lr_prob(texts):
                X = engine.flaky_vectorizer.transform(texts)
                return engine.flaky_models["Logistic Regression"].predict_proba(X)

            lime_exp = explainer.explain_instance_lime(
                custom_text,
                predict_lr_prob,
                num_features=8
            )

        st.markdown("---")
        st.markdown("##### Local Attribution Weights")

        col_l_chart, col_l_table = st.columns([3, 2])
        with col_l_chart:
            fig, ax = plt.subplots(figsize=(7, 4))
            tokens = [x[0] for x in lime_exp]
            weights = [x[1] for x in lime_exp]
            colors = ["#4361EE" if w > 0 else "#E63946" for w in weights]
            ax.barh(tokens[::-1], weights[::-1], color=colors[::-1])
            ax.set_xlabel("Local Feature Weight (LIME)")
            ax.set_title("LIME Token Contributions (Blue = Pushes to Flaky, Red = Pushes to False Positive)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_l_table:
            lime_df = pd.DataFrame([
                {
                    "Token": token,
                    "Weight": f"{w:.4f}",
                    "Influence": "Supports Flaky" if w > 0 else "Supports False Positive"
                }
                for token, w in lime_exp
            ])
            st.dataframe(lime_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Prediction Quadrant Case Studies (Holdout Test Set)")
    q1, q2, q3 = st.columns(3)
    with q1:
        tp_img = os.path.join(XAI_FIGURES_DIR, "lime_true_positive_correctly_flaky.png")
        if os.path.exists(tp_img):
            st.image(tp_img, caption="True Positive: Correctly Diagnosed Flaky", use_container_width=True)
    with q2:
        tn_img = os.path.join(XAI_FIGURES_DIR, "lime_true_negative_correctly_false_positive.png")
        if os.path.exists(tn_img):
            st.image(tn_img, caption="True Negative: Correctly Diagnosed False Positive", use_container_width=True)
    with q3:
        fn_img = os.path.join(XAI_FIGURES_DIR, "lime_false_negative_flaky_missed.png")
        if os.path.exists(fn_img):
            st.image(fn_img, caption="False Negative: Subtle Flakiness Missed", use_container_width=True)

with tab_shift:
    st.subheader("Attribution Stability Under Repository Distribution Shift")
    st.markdown(
        "> **Research Finding:** Comparing feature weights between the model trained with overlapping repositories "
        "and the model trained with strict zero repository overlap."
    )

    shift_img = os.path.join(XAI_FIGURES_DIR, "model_shift_explanation_comparison.png")
    if os.path.exists(shift_img):
        st.image(shift_img, caption="Token Weight Comparison: Random Split vs. Repository-Independent Training", use_container_width=True)

    st.info(
        "**Semantic Invariance:** Core timing and concurrency terms (`flaky`, `timeout`, `race`, `goroutine`, `wait`, `sleep`) "
        "maintain identical positive directional polarity under repository partitioning, confirming that model explanations "
        "are stable and do not rely on spurious repository-specific memorization."
    )
