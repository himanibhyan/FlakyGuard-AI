"""
Page 2: Flaky Test Detector for FlakyGuard AI.
Allows real-time diagnosis of GitHub issues, predicts flakiness and root cause,
and visualizes influential tokens and probability distributions.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.ui_theme import apply_theme
from src.inference import FlakyGuardEngine

st.set_page_config(page_title="Flaky Test Detector | FlakyGuard AI", page_icon="🔍", layout="wide")
apply_theme()

@st.cache_resource
def get_engine():
    return FlakyGuardEngine.get_instance()

engine = get_engine()

st.title("🔍 Flaky Test Detector & Root-Cause Classifier")
st.markdown(
    "Analyze a software issue report to predict whether it represents **genuine test flakiness** "
    "or a **false positive**, diagnose the likely **root cause**, and inspect the influential linguistic features."
)

st.markdown("---")

# Sample selector for easy demonstration
samples = {
    "Custom Input": {"title": "", "desc": "", "lang": "Python"},
    "Sample 1 (Async Race Condition)": {
        "title": "app: TestRemoveUnits is flaky on CI runner",
        "desc": "Tests for Stat() calls fail intermittently with race condition: wait timeout exceeded before goroutine finished channel send.",
        "lang": "Go"
    },
    "Sample 2 (Platform & Time Dependency)": {
        "title": "Date format assertion failure on Windows environment",
        "desc": "Expected timestamp to match local timezone offset, but system clock precision on Windows build causes assertion mismatch every few runs.",
        "lang": "Java"
    },
    "Sample 3 (False Positive - Regular Feature Bug)": {
        "title": "Fix sorting of PatchOperations in UI modal",
        "desc": "The submission form fails to submit when user leaves input field empty. Added null check to prevent crash on submit.",
        "lang": "JavaScript"
    }
}

selected_sample_name = st.selectbox("Quick-fill Example Issue Report (or enter your own):", list(samples.keys()))
default_vals = samples[selected_sample_name]

col_inputs, col_params = st.columns([3, 1])

with col_inputs:
    issue_title = st.text_input(
        "Issue / PR Title:",
        value=default_vals["title"],
        placeholder="e.g., TestAsyncChannelTimeout is intermittently failing"
    )
    issue_desc = st.text_area(
        "Issue Description / Error Output:",
        value=default_vals["desc"],
        height=140,
        placeholder="Paste issue details, stack traces, test logs, or reviewer comments..."
    )

with col_params:
    lang_options = ["Python", "Go", "Java", "JavaScript", "C", "Other"]
    default_lang_idx = lang_options.index(default_vals["lang"]) if default_vals["lang"] in lang_options else 0
    language = st.selectbox("Programming Language:", lang_options, index=default_lang_idx)

    model_choice = st.selectbox(
        "Classification Architecture:",
        ["Random Forest (Best Macro-F1)", "Logistic Regression", "Linear SVM"]
    )
    # Map selection to internal name
    model_name_map = {
        "Random Forest (Best Macro-F1)": "Random Forest",
        "Logistic Regression": "Logistic Regression",
        "Linear SVM": "Linear SVM"
    }
    chosen_model = model_name_map[model_choice]

predict_clicked = st.button("🚀 Analyze Issue Report", type="primary", use_container_width=True)

if predict_clicked:
    if not issue_title.strip() and not issue_desc.strip():
        st.error("Please enter at least an issue title or description.")
    else:
        with st.spinner("Analyzing issue report with trained models..."):
            # Task A Prediction
            res_flaky = engine.predict_flakiness(
                title=issue_title,
                description=issue_desc,
                language=language,
                model_name=chosen_model
            )

            # Task B Prediction (Root Cause)
            res_rc = engine.predict_root_cause(
                title=issue_title,
                description=issue_desc,
                language=language
            )

        st.markdown("---")
        st.subheader("📋 Diagnostic Results")

        # Results Overview Cards
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            if res_flaky["is_flaky"]:
                st.markdown(
                    """
                    <div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; border-radius: 8px; padding: 16px; text-align: center;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: #EF4444; text-transform: uppercase;">Prediction</span>
                        <h2 style="color: #F87171; margin: 4px 0;">FLAKY TEST</h2>
                        <small style="color: #CBD5E1;">Genuine Flakiness (True Positive)</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 8px; padding: 16px; text-align: center;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: #3B82F6; text-transform: uppercase;">Prediction</span>
                        <h2 style="color: #60A5FA; margin: 4px 0;">FALSE POSITIVE</h2>
                        <small style="color: #CBD5E1;">Deterministic Bug or Non-Flaky</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with r2:
            prob_val = res_flaky["flaky_probability"]
            st.metric(
                label="Flakiness Probability (Calibrated)",
                value=f"{prob_val * 100:.1f}%",
                help="Calibrated probability derived from model decision space."
            )

        with r3:
            conf_val = res_flaky["confidence"]
            st.metric(
                label="Prediction Confidence",
                value=f"{conf_val * 100:.1f}%",
                help="Confidence score: max class posterior probability."
            )

        with r4:
            st.metric(
                label="Probable Root Cause",
                value=res_rc["predicted_root_cause"],
                help="Most probable category among the 13 research failure modes."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for detailed inspection
        tab1, tab2, tab3 = st.tabs(["🧬 Root-Cause Analysis", "🔤 Influential Keywords", "📝 Qualitative Interpretation"])

        with tab1:
            col_rc_chart, col_rc_table = st.columns([3, 2])
            with col_rc_chart:
                st.markdown("##### Root-Cause Probability Distribution")
                dist = res_rc["probability_distribution"]
                top_6_dist = dict(list(dist.items())[:6])

                fig, ax = plt.subplots(figsize=(7, 3.8))
                y_pos = range(len(top_6_dist))
                bars = ax.barh(list(top_6_dist.keys())[::-1], [v * 100 for v in list(top_6_dist.values())[::-1]], color="#4361EE")
                ax.set_xlabel("Probability (%)")
                ax.set_xlim(0, max(top_6_dist.values()) * 125)
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_rc_table:
                st.markdown("##### Top Category Rankings")
                rc_table_df = pd.DataFrame([
                    {"Category": k, "Posterior (%)": f"{v * 100:.2f}%"}
                    for k, v in list(dist.items())[:6]
                ])
                st.dataframe(rc_table_df, hide_index=True, use_container_width=True)

        with tab2:
            st.markdown("##### Most Influential Words in Input Report")
            words_data = res_flaky["influential_words"]
            if words_data:
                col_w_chart, col_w_table = st.columns([3, 2])
                with col_w_chart:
                    fig, ax = plt.subplots(figsize=(7, 3.8))
                    tokens = [w["word"] for w in words_data[:8]]
                    weights = [w["contribution"] for w in words_data[:8]]
                    colors = ["#4361EE" if c > 0 else "#E63946" for c in weights]
                    ax.barh(tokens[::-1], weights[::-1], color=colors[::-1])
                    ax.set_xlabel("Contribution (TF-IDF × Model Weight)")
                    ax.set_title("Top Word Influences (Blue: Supports Flaky, Red: Supports False Positive)", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_w_table:
                    words_df = pd.DataFrame(words_data)
                    st.dataframe(words_df, hide_index=True, use_container_width=True)
            else:
                st.info("Input tokens did not match active vocabulary features in the TF-IDF space.")

        with tab3:
            st.markdown("##### Diagnostic Narrative")
            if res_flaky["is_flaky"]:
                st.markdown(
                    f"- The issue report **strongly indicates test flakiness** with a calibrated probability of **{res_flaky['flaky_probability']*100:.1f}%**.\n"
                    f"- The leading suspected failure mechanism is **`{res_rc['predicted_root_cause']}`** ({res_rc['confidence']*100:.1f}% confidence).\n"
                    f"- **Recommended CI/CD Action:** Quarantine the identified test target, inspect timing delays or shared state, and check for asynchronous race conditions."
                )
            else:
                st.markdown(
                    f"- The issue report is classified as a **False Positive (Deterministic Issue)** with **{res_flaky['false_positive_probability']*100:.1f}%** confidence.\n"
                    "- The report exhibits vocabulary characteristic of conventional bugs or deterministic test failures rather than stochastic flakiness.\n"
                    "- **Recommended Action:** Treat as a standard deterministic defect; do not apply flakiness mitigation loops."
                )
