"""
Page 1: Overview Dashboard for FlakyGuard AI.
Visualizes dataset statistics, label proportions, root-cause prevalence,
and model benchmark comparisons using actual computed figures.
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.ui_theme import apply_theme, render_metric_card

st.set_page_config(page_title="Overview Dashboard | FlakyGuard AI", page_icon="📊", layout="wide")
apply_theme()

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "model_comparison.json")
PREPROC_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "preprocessing_summary.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

with open(PREPROC_PATH) as f:
    preproc_data = json.load(f)
with open(METRICS_PATH) as f:
    model_data = json.load(f)

full_df = pd.read_csv(os.path.join(PROCESSED_DIR, "full_cleaned_dataset.csv"))
rc_df = pd.read_csv(os.path.join(PROCESSED_DIR, "root_cause_dataset.csv"))

st.title("📊 Dataset Overview & Research Dashboard")
st.caption("Empirical distributions and model benchmarks from Barbosa et al., IEEE TSE 2023")

# Top KPI metrics
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Total Dataset Issues", f"{preproc_data['deduplicated_records']:,}", "1,543 raw GitHub records")
with c2:
    render_metric_card("Languages Studied", "5", "C, Go, Java, JS, Python")
with c3:
    render_metric_card("Confirmed Flaky (T)", f"{preproc_data['task_a_binary']['flaky_positive (T)']}", "63.7% of binary subset")
with c4:
    render_metric_card("Root-Cause Classes", f"{preproc_data['task_b_root_cause']['num_classes']}", "589 confirmed causes")
with c5:
    best_f1 = model_data["task_a_flakiness_detection"]["models"]["Random Forest"]["macro_f1"]
    render_metric_card("Best Model Macro-F1", f"{best_f1:.3f}", "Task A Random Forest")

st.markdown("---")

# Charts row 1
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🏷️ Flakiness Label Distribution")
    status_counts = full_df["flakiness_status"].value_counts(dropna=False)
    status_df = pd.DataFrame({
        "Status": status_counts.index.map({"T": "Flaky (T)", "F": "False Positive (F)", "ND": "Not Determined (ND)"}),
        "Issues": status_counts.values
    }).dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=status_df, x="Status", y="Issues", palette=["#4361EE", "#E63946", "#F77F00"], ax=ax)
    ax.set_ylabel("Issue Count")
    ax.set_title("Flakiness Class Proportions in TSE22", weight="bold")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)} ({h/status_counts.sum()*100:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                    ha="center", va="center", color="white", weight="bold", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.markdown("### 🌐 Cross-Language Flakiness Breakdown")
    fig_lang = os.path.join(FIGURES_DIR, "cross_language_flakiness.png")
    if os.path.exists(fig_lang):
        st.image(fig_lang, caption="Flakiness Distribution Across C, Go, Java, JavaScript, Python", use_container_width=True)

st.markdown("---")

# Charts row 2
col_rc, col_models = st.columns(2)

with col_rc:
    st.markdown("### 🧬 Root-Cause Category Distribution (N=589)")
    fig_rc = os.path.join(FIGURES_DIR, "root_cause_distribution.png")
    if os.path.exists(fig_rc):
        st.image(fig_rc, caption="Prevalence of 13 Root-Cause Categories in Confirmed Flaky Tests", use_container_width=True)

with col_models:
    st.markdown("### 🏆 Classical ML Model Benchmark Comparison")
    task_a_models = model_data["task_a_flakiness_detection"]["models"]
    m_names = list(task_a_models.keys())
    m_macro_f1 = [task_a_models[m]["macro_f1"] for m in m_names]
    m_acc = [task_a_models[m]["accuracy"] for m in m_names]
    m_roc = [task_a_models[m]["roc_auc"] for m in m_names]

    bench_df = pd.DataFrame({
        "Model": m_names,
        "Macro-F1": m_macro_f1,
        "Accuracy": m_acc,
        "ROC-AUC": m_roc
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(m_names))
    w = 0.25
    ax.bar([i - w for i in x], m_macro_f1, width=w, label="Macro-F1", color="#4361EE")
    ax.bar(x, m_acc, width=w, label="Accuracy", color="#3A0CA3")
    ax.bar([i + w for i in x], m_roc, width=w, label="ROC-AUC", color="#4CC9F0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(m_names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Task A Model Performance Benchmark", weight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Dataset Insights Section
st.markdown("### 💡 Empirical Dataset Insights")
c_in1, c_in2, c_in3 = st.columns(3)

with c_in1:
    st.info(
        "**Asynchronous Waiting & Concurrency Dominate:**\n\n"
        "Together, `Async Wait` (17.5%) and `Concurrency` (16.5%) account for over one-third of all confirmed flaky tests, "
        "highlighting non-deterministic timing as the single largest failure vector."
    )

with c_in2:
    st.warning(
        "**Significant Class Imbalance:**\n\n"
        "True flaky issues (701) outnumber false positive issues (400) by a ratio of 1.75:1. "
        "Models must be evaluated using Macro-F1 and PR-AUC rather than raw Accuracy to prevent majority-class bias."
    )

with c_in3:
    st.success(
        "**Cross-Language Differences:**\n\n"
        "Go and Java exhibit higher proportions of concurrency and async wait issues, whereas Python and JavaScript "
        "have higher rates of platform dependency and resource leak problems."
    )
