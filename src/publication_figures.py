"""
Publication Figures Generator for FlakyGuard AI Research Paper.
Generates 10 high-resolution, publication-quality figures under:
artifacts/figures/publication/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

FIGURES_PUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures", "publication"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

# Set publication style
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PALETTE = {
    "primary": "#1D3557",
    "secondary": "#457B9D",
    "accent": "#E63946",
    "light": "#A8DADC",
    "neutral": "#F1FAEE",
    "dark": "#2B2D42",
    "rf": "#2A9D8F",
    "lr": "#457B9D",
    "svm": "#E76F51"
}


def make_dir():
    os.makedirs(FIGURES_PUB_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# FIGURE 1: System Architecture Diagram
# -----------------------------------------------------------------------------
def generate_fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color, text_color="white"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=color, edgecolor="#2B2D42", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=10.5, weight="bold", color=text_color)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center",
                fontsize=8.5, color=text_color)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#2B2D42", edgecolor="#2B2D42", width=1.5, headwidth=7),
            ha="center", va="bottom", fontsize=8, weight="bold"
        )

    # 1. Ingestion
    draw_box(0.05, 0.65, 0.22, 0.25, "1. Issue Ingestion", "GitHub Issues (N=1,101)\nC, Go, Java, JS, Python\nOwner/Repo Extraction", "#457B9D")
    
    # 2. Preprocessing
    draw_box(0.38, 0.65, 0.24, 0.25, "2. Preprocessing", "Noise & URL Stripping\nCanonical Repository Mapping\nStrict Partition Isolation", "#2A9D8F")
    
    # 3. Representation
    draw_box(0.72, 0.65, 0.23, 0.25, "3. Feature Space", "Word-level TF-IDF (8k feats)\nSublinear Term Frequency\nFitted Only on Train Splits", "#1D3557")

    draw_arrow(0.27, 0.775, 0.38, 0.775)
    draw_arrow(0.62, 0.775, 0.72, 0.775)

    # 4. Modeling & Auditing
    draw_box(0.05, 0.15, 0.35, 0.35, "4. Multi-Regime Modeling & Auditing", 
             "- Task A: Binary Flakiness Detection\n  (Random Forest, Logistic Regression, Linear SVM)\n- Task B: Root Cause Classification (13 & 4 classes)\n- 10-Seed GroupShuffleSplit (Zero Overlap)\n- Leave-One-Language-Out (5x3 Matrix)\n- Cluster Bootstrap (B=1,000)", "#2B2D42")

    # 5. Explainability
    draw_box(0.48, 0.15, 0.22, 0.35, "5. Explainable AI", 
             "- Global SHAP LinearExplainer\n- Tree MDI Impurity Tracking\n- Local LIME Word Attributions\n- Distribution Shift Sign Audit", "#E76F51")

    # 6. Deployment / Actionability
    draw_box(0.78, 0.15, 0.18, 0.35, "6. Triage Delivery", 
             "- CI/CD Fast Triage\n- High-Level Failure Router\n- Local Debugging Insights\n- Streamlit Dashboard", "#6A4C93")

    draw_arrow(0.835, 0.65, 0.835, 0.50)
    draw_arrow(0.72, 0.325, 0.48, 0.325)
    draw_arrow(0.40, 0.325, 0.48, 0.325)
    draw_arrow(0.70, 0.325, 0.78, 0.325)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.05, 1.0)
    plt.title("FlakyGuard AI: System Architecture & Research Validation Flow", weight="bold", fontsize=13, pad=15)
    
    out_path = os.path.join(FIGURES_PUB_DIR, "system_architecture.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 1] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 2: Dataset Distribution
# -----------------------------------------------------------------------------
def generate_fig2_dataset():
    bin_path = os.path.join(PROCESSED_DIR, "flaky_binary_dataset.csv")
    rc_path = os.path.join(PROCESSED_DIR, "root_cause_dataset.csv")
    
    bin_df = pd.read_csv(bin_path)
    rc_df = pd.read_csv(rc_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: Binary Labels
    label_counts = bin_df["label"].value_counts().rename({1: "Flaky (T)", 0: "False Positive (F)"})
    colors_a = ["#2A9D8F", "#E76F51"]
    bars_a = axes[0].bar(label_counts.index, label_counts.values, color=colors_a, edgecolor="black", width=0.55)
    axes[0].set_title("(a) Binary Flakiness Distribution\n(Total N = 1,101)", weight="bold")
    axes[0].set_ylabel("Number of Issue Reports")
    axes[0].set_ylim(0, 850)
    for b in bars_a:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h + 15, f"{h}\n({h/1101*100:.1f}%)", ha="center", fontsize=9, weight="bold")

    # Panel B: Languages
    lang_counts = bin_df["language"].value_counts()
    colors_b = ["#457B9D", "#1D3557", "#2A9D8F", "#E9C46A", "#F4A261"]
    bars_b = axes[1].bar(lang_counts.index.str.upper(), lang_counts.values, color=colors_b, edgecolor="black", width=0.6)
    axes[1].set_title("(b) Target Language Representation\n(5 Software Ecosystems)", weight="bold")
    axes[1].set_ylabel("Issue Reports")
    axes[1].set_ylim(0, 310)
    for b in bars_b:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, h + 6, f"{h}", ha="center", fontsize=9, weight="bold")

    # Panel C: Top Root Causes
    rc_counts = rc_df["root_cause_category"].value_counts().head(7)
    bars_c = axes[2].barh(rc_counts.index[::-1], rc_counts.values[::-1], color="#2B2D42", edgecolor="black", height=0.6)
    axes[2].set_title("(c) Top-7 Root-Cause Categories\n(Confirmed Flaky N = 589)", weight="bold")
    axes[2].set_xlabel("Number of Issues")
    axes[2].set_xlim(0, 125)
    for b in bars_c:
        w = b.get_width()
        axes[2].text(w + 2, b.get_y() + b.get_height()/2, f"{int(w)}", va="center", fontsize=8.5, weight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "dataset_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 2] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 3: Binary Classification Model Comparison
# -----------------------------------------------------------------------------
def generate_fig3_binary_models():
    # Stratified Random 80/20 Holdout (N=221)
    # RF: Macro-F1 0.6848 (CI: 0.6235, 0.7466), Acc 0.7240, ROC-AUC 0.7310 (CI: 0.6601, 0.8019), PR-AUC 0.8258
    # LR: Macro-F1 0.6555 (CI: 0.5946, 0.7198), Acc 0.6923, ROC-AUC 0.7262 (CI: 0.6555, 0.7978), PR-AUC 0.8066
    # SVM: Macro-F1 0.6107 (CI: 0.5400, 0.6757), Acc 0.6878, ROC-AUC 0.6993 (CI: 0.6309, 0.7618), PR-AUC 0.8035
    metrics = ["Macro-F1", "Accuracy", "ROC-AUC", "PR-AUC"]
    models = ["Random Forest", "Logistic Regression", "Linear SVM"]
    
    data = np.array([
        [0.6848, 0.7240, 0.7310, 0.8258],
        [0.6555, 0.6923, 0.7262, 0.8066],
        [0.6107, 0.6878, 0.6993, 0.8035]
    ])

    # 95% CI error bounds for Macro-F1 and ROC-AUC
    # [ [yerr_lo], [yerr_hi] ]
    # RF Macro-F1: 0.6848 - 0.6235 = 0.0613, 0.7466 - 0.6848 = 0.0618
    # LR Macro-F1: 0.6555 - 0.5946 = 0.0609, 0.7198 - 0.6555 = 0.0643
    # SVM Macro-F1: 0.6107 - 0.5400 = 0.0707, 0.6757 - 0.6107 = 0.0650

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.25

    colors = [PALETTE["rf"], PALETTE["lr"], PALETTE["svm"]]

    for i, m_name in enumerate(models):
        bars = ax.bar(x + (i - 1) * width, data[i], width=width, label=m_name, color=colors[i], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=8, weight="bold")

    # Add baseline indicator (chance Macro-F1 under empirical prior = 0.455)
    ax.axhline(0.455, color="#E63946", linestyle="--", linewidth=1.2, label="Majority Class Prior Baseline (0.455)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, weight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Task A: Binary Flakiness Detection Performance (Stratified Random Holdout N=221)", weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "binary_model_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 3] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 4: Repository-Independent Comparison (Holdout vs Repo-Indep vs 10-Partition Mean)
# -----------------------------------------------------------------------------
def generate_fig4_repo_independent():
    # Random Split vs Single Repo-Independent vs 10-Partition Mean Macro-F1
    # Random Forest: 0.6848 -> 0.7092 (Seed 42) -> 0.6542 (10-seed mean)
    # Logistic Regression: 0.6555 -> 0.6978 (Seed 42) -> 0.6670 (10-seed mean)
    # Linear SVM: 0.6107 -> 0.5321 (Seed 42) -> 0.5201 (10-seed mean)

    regimes = ["Random Split (Holdout)", "Repo-Independent (Seed 42)", "Repo-Independent (10-Partition Mean)"]
    models = ["Random Forest", "Logistic Regression", "Linear SVM"]

    scores = np.array([
        [0.6848, 0.7092, 0.6542],  # RF
        [0.6555, 0.6978, 0.6670],  # LR
        [0.6107, 0.5321, 0.5201]   # SVM
    ])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(regimes))
    width = 0.24
    colors = [PALETTE["rf"], PALETTE["lr"], PALETTE["svm"]]

    for i, m_name in enumerate(models):
        bars = ax.bar(x + (i - 1) * width, scores[i], width=width, label=m_name, color=colors[i], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=8.5, weight="bold")

    ax.axhline(0.455, color="#E63946", linestyle="--", linewidth=1.2, label="Majority Prior Baseline (0.455)")

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, weight="bold")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Generalization Under Cross-Repository Shift (Macro-F1 Comparison)", weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "repository_independent_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 4] Saved: {out_path}")



# -----------------------------------------------------------------------------
# FIGURE 5: LOLO Macro-F1 Comparison Across Languages
# -----------------------------------------------------------------------------
def generate_fig5_lolo_comparison():
    langs = ["C", "Go", "Java", "JavaScript", "Python", "POOLED MEAN"]
    # Scores from second-level audit:
    rf_scores = [0.5931, 0.6298, 0.4922, 0.6929, 0.5931, 0.6002]
    lr_scores = [0.6635, 0.6327, 0.5144, 0.6863, 0.5297, 0.6053]
    svm_scores = [0.4974, 0.5365, 0.4691, 0.5263, 0.3816, 0.4822]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(langs))
    width = 0.25

    b1 = ax.bar(x - width, rf_scores, width=width, label="Random Forest", color=PALETTE["rf"], edgecolor="black")
    b2 = ax.bar(x, lr_scores, width=width, label="Logistic Regression", color=PALETTE["lr"], edgecolor="black")
    b3 = ax.bar(x + width, svm_scores, width=width, label="Linear SVM", color=PALETTE["svm"], edgecolor="black")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}", ha="center", fontsize=7.5, weight="bold")

    ax.axvline(4.5, color="#2B2D42", linestyle=":", linewidth=1.5)
    ax.axhline(0.455, color="#E63946", linestyle="--", linewidth=1.2, label="Majority Class Prior Baseline (0.455)")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, weight="bold")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Leave-One-Language-Out (LOLO) Cross-Language Generalization (Unbiased 5x3 Evaluation)", weight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "lolo_language_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 5] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 6: Repeated Repository Split Performance Distribution
# -----------------------------------------------------------------------------
def generate_fig6_repeated_repo_dist():
    csv_path = os.path.join(METRICS_DIR, "repeated_repository_generalization.csv")
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: Macro-F1 across 10 repeated partitions
    sns.boxplot(data=df, x="model", y="macro_f1", hue="model", ax=axes[0], palette=[PALETTE["rf"], PALETTE["lr"], PALETTE["svm"]], width=0.45, legend=False)
    sns.stripplot(data=df, x="model", y="macro_f1", ax=axes[0], color="black", size=6, jitter=0.15, alpha=0.8)
    axes[0].set_title("(a) Macro-F1 Across 10 Repeated Partitions\n(Zero Repository Overlap per Partition)", weight="bold")
    axes[0].set_ylabel("Macro-F1 Score")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0.40, 0.75)
    axes[0].axhline(0.455, color="#E63946", linestyle="--", linewidth=1.2, label="Chance Prior")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: ROC-AUC across 10 repeated partitions
    sns.boxplot(data=df, x="model", y="roc_auc", hue="model", ax=axes[1], palette=[PALETTE["rf"], PALETTE["lr"], PALETTE["svm"]], width=0.45, legend=False)
    sns.stripplot(data=df, x="model", y="roc_auc", ax=axes[1], color="black", size=6, jitter=0.15, alpha=0.8)
    axes[1].set_title("(b) ROC-AUC Across 10 Repeated Partitions\n(Zero Repository Overlap per Partition)", weight="bold")
    axes[1].set_ylabel("ROC-AUC Score")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0.60, 0.82)
    axes[1].axhline(0.50, color="#E63946", linestyle="--", linewidth=1.2, label="Random Guessing")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "repeated_repository_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 6] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 7: Root-Cause Classification Confusion Matrix
# -----------------------------------------------------------------------------
def generate_fig7_rc_confusion_matrix():
    # Use saved cm for Task B or generate high-res plot
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix
    from src.feature_engineering import get_tfidf_vectorizer

    rc_path = os.path.join(PROCESSED_DIR, "root_cause_dataset.csv")
    rc_df = pd.read_csv(rc_path)

    le = LabelEncoder()
    y_encoded = le.fit_transform(rc_df["root_cause_category"])
    class_names = list(le.classes_)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        rc_df["combined_text"], y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )
    vec = get_tfidf_vectorizer(max_features=8000, min_df=2)
    X_tr = vec.fit_transform(X_tr_raw)
    X_te = vec.transform(X_te_raw)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=150, max_depth=25, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="gray")

    ax.set_title("13-Class Root-Cause Confusion Matrix (Random Forest, Test N=118)", weight="bold", pad=12)
    ax.set_xlabel("Predicted Root Cause Category", weight="bold")
    ax.set_ylabel("True Root Cause Category", weight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "root_cause_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 7] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 8: Root-Cause 13-Class vs. 4-Class Comparison with Rigorous Metric Baselines
# -----------------------------------------------------------------------------
def generate_fig8_root_cause_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    categories = ["13 Empirical Classes", "4 Macroscopic Classes"]
    x = np.arange(len(categories))
    width = 0.26

    # PANEL A: Macro-F1 vs Macro-F1 Baselines
    # Model Macro-F1: 13-class RF = 0.2834, 4-class LR = 0.4222
    # Uniform Random Macro-F1: 13-class = 0.0655, 4-class = 0.2383
    # Majority Prior Macro-F1: 13-class = 0.0232, 4-class = 0.1358
    model_mf1 = [0.2834, 0.4222]
    unif_mf1 = [0.0655, 0.2383]
    maj_mf1 = [0.0232, 0.1358]

    b1 = axes[0].bar(x - width, model_mf1, width=width, label="Model Macro-F1", color=PALETTE["rf"], edgecolor="black")
    b2 = axes[0].bar(x, unif_mf1, width=width, label="Uniform Random Macro-F1 (Expected)", color="#A8DADC", edgecolor="black", hatch="//")
    b3 = axes[0].bar(x + width, maj_mf1, width=width, label="Majority Class Macro-F1", color="#E63946", edgecolor="black", hatch="\\\\")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}", ha="center", fontsize=8.5, weight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, weight="bold", fontsize=9.5)
    axes[0].set_ylabel("Macro-F1 Score")
    axes[0].set_ylim(0, 0.52)
    axes[0].set_title("(a) Macro-F1 vs. Empirical Macro-F1 Baselines", weight="bold")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # PANEL B: Accuracy vs Accuracy Baselines
    # Model Accuracy: 13-class RF = 0.3729, 4-class LR = 0.4831
    # Uniform Random Accuracy: 13-class = 0.0776, 4-class = 0.2507
    # Majority Prior Accuracy: 13-class = 0.1780, 4-class = 0.3729
    model_acc = [0.3729, 0.4831]
    unif_acc = [0.0776, 0.2507]
    maj_acc = [0.1780, 0.3729]

    b4 = axes[1].bar(x - width, model_acc, width=width, label="Model Accuracy", color=PALETTE["primary"], edgecolor="black")
    b5 = axes[1].bar(x, unif_acc, width=width, label="Uniform Random Accuracy (1/K)", color="#A8DADC", edgecolor="black", hatch="//")
    b6 = axes[1].bar(x + width, maj_acc, width=width, label="Majority Class Accuracy", color="#E63946", edgecolor="black", hatch="\\\\")

    for bars in [b4, b5, b6]:
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.1%}", ha="center", fontsize=8.5, weight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, weight="bold", fontsize=9.5)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 0.65)
    axes[1].set_title("(b) Accuracy vs. Empirical Accuracy Baselines", weight="bold")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8.5)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Root-Cause Classification: 13-Class vs. 4-Class Performance Deconstructed by Metric", weight="bold", fontsize=11.5, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "root_cause_13_vs_4_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 8] Saved: {out_path}")



# -----------------------------------------------------------------------------
# FIGURE 9: SHAP Global Explanation Figure
# -----------------------------------------------------------------------------
def generate_fig9_shap_global():
    import joblib
    import shap
    from sklearn.model_selection import train_test_split

    bin_path = os.path.join(PROCESSED_DIR, "flaky_binary_dataset.csv")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    bin_df = pd.read_csv(bin_path)
    vec = joblib.load(os.path.join(models_dir, "flaky_vectorizer.joblib"))
    model_lr = joblib.load(os.path.join(models_dir, "flaky_logistic_regression.joblib"))

    _, X_test_raw, _, _ = train_test_split(
        bin_df["combined_text"], bin_df["label"].values, test_size=0.20, stratify=bin_df["label"].values, random_state=42
    )

    X_shap = vec.transform(X_test_raw.iloc[:120])
    explainer = shap.LinearExplainer(model_lr, masker=shap.maskers.Independent(data=X_shap))
    shap_vals = explainer(X_shap)

    feature_names = np.array(vec.get_feature_names_out())
    mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-15:]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(feature_names[top_idx], mean_abs_shap[top_idx], color=PALETTE["secondary"], edgecolor="black", height=0.65)
    
    ax.set_title("SHAP Global Feature Importance (Top 15 Tokens, Logistic Regression)", weight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Flakiness Prediction Output)", weight="bold")
    ax.set_xlim(0, max(mean_abs_shap[top_idx]) * 1.15)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "shap_global_explanation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 9] Saved: {out_path}")


# -----------------------------------------------------------------------------
# FIGURE 10: Explainability Attribution Stability (Random Split vs. Repo-Independent)
# -----------------------------------------------------------------------------
def generate_fig10_explainability_stability():
    tokens = ["flaky", "fix", "race", "timeout", "intermittent", "sleep", "add", "wait", "error", "goroutine"]
    w_rand = [1.8239, 0.3755, 0.4290, 0.4180, 0.2476, 0.0478, -0.5960, -0.1379, -0.0695, -0.1708]
    w_repo = [1.7810, 0.7610, 0.4961, 0.3593, 0.1712, 0.0606, -0.5555, -0.1718, -0.1940, 0.0387]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(tokens))
    width = 0.35

    b1 = ax.bar(x - width/2, w_rand, width=width, label="Random Split Model (With Repo Overlap)", color=PALETTE["primary"], edgecolor="black")
    b2 = ax.bar(x + width/2, w_repo, width=width, label="Repo-Independent Model (Zero Overlap)", color=PALETTE["accent"], edgecolor="black")

    # Highlight goroutine sign flip
    ax.annotate("Sign Inversion\n(-0.17 -> +0.04)", xy=(9 + width/2, 0.04), xytext=(8.0, 0.55),
                arrowprops=dict(facecolor="#2B2D42", shrink=0.08, width=1.5, headwidth=6),
                fontsize=8.5, weight="bold", color="#E63946")

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, weight="bold", fontsize=9.5)
    ax.set_ylabel("Logistic Regression Feature Weight (Coefficient)")
    ax.set_ylim(-0.8, 2.1)
    ax.set_title("Attribution Stability Audit: Random Split vs. Repository-Independent Model", weight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_PUB_DIR, "explainability_stability.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Publication Figure 10] Saved: {out_path}")


def main():
    make_dir()
    print("=" * 65)
    print("GENERATING 10 PUBLICATION-QUALITY FIGURES")
    print("=" * 65)
    generate_fig1_architecture()
    generate_fig2_dataset()
    generate_fig3_binary_models()
    generate_fig4_repo_independent()
    generate_fig5_lolo_comparison()
    generate_fig6_repeated_repo_dist()
    generate_fig7_rc_confusion_matrix()
    generate_fig8_root_cause_comparison()
    generate_fig9_shap_global()
    generate_fig10_explainability_stability()
    print("=" * 65)
    print("ALL 10 FIGURES GENERATED SUCCESSFULLY IN:", FIGURES_PUB_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()
