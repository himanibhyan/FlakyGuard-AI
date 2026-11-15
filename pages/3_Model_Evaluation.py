"""
Page 3: Model Evaluation for FlakyGuard AI.
Displays comprehensive performance benchmarks, classification reports,
confusion matrices, ROC/PR curve comparisons, repository-independent validation,
and Leave-One-Language-Out (LOLO) cross-language generalizability.
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.ui_theme import apply_theme

st.set_page_config(page_title="Model Evaluation | FlakyGuard AI", page_icon="📈", layout="wide")
apply_theme()

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "model_comparison.json")
REPO_GEN_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "repository_generalization_results.csv")
LOLO_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "lolo_results.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures")

with open(METRICS_PATH) as f:
    benchmark_data = json.load(f)

st.title("📈 Model Evaluation & Empirical Benchmarks")
st.caption("Rigorous evaluation on holdout test partitions, unseen software repositories, and cross-language domain shifts.")

st.markdown("---")

task_choice = st.radio(
    "Select Evaluation Dimension:",
    [
        "Task A: Flakiness Detection (Random 80/20 Baseline)",
        "Repository-Independent Generalization (Zero Repo Overlap)",
        "Leave-One-Language-Out (LOLO) Cross-Language",
        "Task B: Root-Cause Classification (13 Classes)"
    ],
    horizontal=True
)

st.markdown("---")

if "Random 80/20 Baseline" in task_choice:
    st.subheader("Task A: Flakiness Detection Baseline (Stratified Random 80/20 Split)")
    task_a = benchmark_data["task_a_flakiness_detection"]["models"]

    rows = []
    for model_name, m in task_a.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "Macro-F1": f"{m['macro_f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}" if isinstance(m['roc_auc'], (int, float)) else m['roc_auc'],
            "PR-AUC": f"{m['pr_auc']:.4f}" if isinstance(m['pr_auc'], (int, float)) else m['pr_auc']
        })

    eval_df = pd.DataFrame(rows)
    st.dataframe(eval_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.markdown("##### Receiver Operating Characteristic (ROC) Curves")
        roc_img = os.path.join(FIGURES_DIR, "task_a_roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)

    with col_pr:
        st.markdown("##### Precision-Recall (PR) Curves")
        pr_img = os.path.join(FIGURES_DIR, "task_a_pr_curves.png")
        if os.path.exists(pr_img):
            st.image(pr_img, use_container_width=True)

    st.markdown("---")

    st.markdown("##### Confusion Matrix Inspection")
    cm_model = st.selectbox("Select Model to Inspect Confusion Matrix:", list(task_a.keys()))
    cm_filename = f"cm_task_a_{cm_model.lower().replace(' ', '_')}.png"
    cm_path = os.path.join(FIGURES_DIR, cm_filename)

    c_cm_img, c_cm_report = st.columns([1, 1])
    with c_cm_img:
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix for {cm_model}", use_container_width=True)

    with c_cm_report:
        st.markdown(f"**Classification Report ({cm_model}):**")
        rep = task_a[cm_model]["classification_report"]
        rep_rows = []
        for target in ["False Positive", "Flaky"]:
            if target in rep:
                rep_rows.append({
                    "Class": target,
                    "Precision": f"{rep[target]['precision']:.4f}",
                    "Recall": f"{rep[target]['recall']:.4f}",
                    "F1-Score": f"{rep[target]['f1-score']:.4f}",
                    "Support": rep[target]["support"]
                })
        st.dataframe(pd.DataFrame(rep_rows), hide_index=True, use_container_width=True)
        st.markdown(
            f"- **Macro Average F1:** `{rep['macro avg']['f1-score']:.4f}`\n"
            f"- **Weighted Average F1:** `{rep['weighted avg']['f1-score']:.4f}`\n"
            f"- **Overall Accuracy:** `{rep['accuracy']:.4f}`"
        )

elif "Repository-Independent" in task_choice:
    st.subheader("Repository-Independent Generalization (Zero Repository Overlap)")
    st.markdown(
        "> **Research Question:** Does flaky test detection performance degrade when testing on **completely unseen software repositories**?  \n"
        "> Using `GroupShuffleSplit`, samples are partitioned by canonical repository ID such that **0 repositories overlap** between training (512 repos, 874 issues) and testing (129 repos, 227 issues)."
    )

    if os.path.exists(REPO_GEN_PATH):
        repo_df = pd.read_csv(REPO_GEN_PATH)
        display_repo_df = repo_df[[
            "model", "accuracy", "precision", "recall", "f1_score", "macro_f1", "roc_auc",
            "baseline_macro_f1", "macro_f1_gap"
        ]].copy()
        display_repo_df.columns = [
            "Model", "Accuracy", "Precision", "Recall", "F1", "Macro-F1 (Repo-Indep)", "ROC-AUC",
            "Random Split Macro-F1", "Macro-F1 Gap (Δ)"
        ]
        st.dataframe(display_repo_df, hide_index=True, use_container_width=True)

    col_fig, col_notes = st.columns([3, 2])
    with col_fig:
        comp_img = os.path.join(FIGURES_DIR, "repository_generalization_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Macro-F1 Comparison: Random Split vs. Strict Zero-Overlap Repository Split", use_container_width=True)

    with col_notes:
        st.markdown("##### Key Generalization Findings:")
        st.info(
            "**Robust Generalization Maintained:**\n\n"
            "- **Random Forest** achieved **0.7092 Macro-F1** (74.0% accuracy) on unseen repositories, confirming that performance is driven by genuine cross-project flakiness semantics rather than repository memorization.\n"
            "- **Logistic Regression** achieved **0.6978 Macro-F1** (ROC-AUC: 0.7727).\n"
            "- **Linear SVM** demonstrated sensitivity to threshold shifts, indicating that probability calibration is vital under distribution shift."
        )

elif "Leave-One-Language-Out" in task_choice:
    st.subheader("Leave-One-Language-Out (LOLO) Cross-Language Generalization")
    st.markdown(
        "> **Research Question:** Can a model trained on four programming languages generalize to an unseen fifth language?  \n"
        "> Models are trained exclusively on the 4 other languages with TF-IDF vocabulary fitted strictly on training data."
    )

    if os.path.exists(LOLO_PATH):
        lolo_df = pd.read_csv(LOLO_PATH)
        display_lolo = lolo_df[[
            "held_out_language", "model", "test_samples", "test_flaky_ratio", "accuracy", "precision", "recall", "macro_f1", "roc_auc"
        ]].copy()
        display_lolo.columns = [
            "Held-Out Target Language", "Architecture", "Test Samples", "Flaky Test Ratio", "Accuracy", "Precision", "Recall", "Macro-F1", "ROC-AUC"
        ]
        st.dataframe(display_lolo, hide_index=True, use_container_width=True)

    lolo_chart = os.path.join(FIGURES_DIR, "lolo_language_comparison.png")
    if os.path.exists(lolo_chart):
        st.image(lolo_chart, caption="LOLO Macro-F1 Benchmark Across C, Go, Java, JavaScript, Python", use_container_width=True)

    st.markdown("---")
    st.markdown("##### Cross-Language Transfer Insights:")
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            "**High Transferability Languages:**\n\n"
            "- **JavaScript (`js`)**: Highest cross-language generalization (Random Forest Macro-F1: 0.6929, ROC-AUC: 0.7636; LR Macro-F1: 0.6863).\n"
            "- **Go (`go`)**: Strong transferability (RF Macro-F1: 0.6298, LR Macro-F1: 0.6327)."
        )
    with c2:
        st.warning(
            "**Challenging Transfer Languages:**\n\n"
            "- **Java (`java`) & Python (`python`)**: Models exhibit lower cross-language F1 (0.49-0.59) due to domain vocabulary drift and high recall bias towards majority flaky tests."
        )

else:
    st.subheader("Task B: Root-Cause Multi-Class Benchmark (13 Categories)")
    task_b = benchmark_data["task_b_root_cause_classification"]["models"]
    classes = benchmark_data["task_b_root_cause_classification"]["classes"]

    rows_b = []
    for model_name, m in task_b.items():
        rows_b.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro-F1": f"{m['macro_f1']:.4f}",
            "Weighted-F1": f"{m['weighted_f1']:.4f}"
        })
    st.dataframe(pd.DataFrame(rows_b), hide_index=True, use_container_width=True)

    st.markdown("---")

    cm_model_b = st.selectbox("Inspect Root-Cause Confusion Matrix:", list(task_b.keys()))
    cm_b_filename = f"cm_task_b_{cm_model_b.lower().replace(' ', '_')}.png"
    cm_b_path = os.path.join(FIGURES_DIR, cm_b_filename)

    if os.path.exists(cm_b_path):
        st.image(cm_b_path, caption=f"Root-Cause Multi-Class Confusion Matrix ({cm_model_b})", use_container_width=True)

    st.markdown("---")

    st.markdown("##### Per-Category Precision, Recall, and F1 Breakdown")
    per_class_data = task_b[cm_model_b]["per_class"]
    pc_rows = []
    for cls_name, vals in per_class_data.items():
        pc_rows.append({
            "Category": cls_name,
            "Precision": f"{vals['precision']:.4f}",
            "Recall": f"{vals['recall']:.4f}",
            "F1-Score": f"{vals['f1-score']:.4f}",
            "Holdout Support": vals["support"]
        })
    st.dataframe(pd.DataFrame(pc_rows), hide_index=True, use_container_width=True)
