"""
Data Preprocessing Module for FlakyGuard AI.
Implements robust data cleaning, duplicate removal, text normalization,
leakage-free feature preparation, and dataset export for Task A and Task B.
"""

import os
import re
import json
import pandas as pd
import numpy as np

RAW_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_flaky_issues.csv"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def clean_text(text: str) -> str:
    """
    Normalizes textual content:
    - Strips URL links
    - Removes code block decorators and markdown noise
    - Collapses repetitive whitespace
    - Lowercases text
    """
    if not isinstance(text, str) or text.strip() in ["", "--", "nan", "None"]:
        return ""

    # Replace URLs with whitespace
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Strip markdown backticks, hashtags, blockquotes
    text = re.sub(r"[`#*_\>~]", " ", text)
    # Replace markdown table pipes, dashes
    text = re.sub(r"\||\-{2,}", " ", text)
    # Strip control / special characters but keep punctuation useful for NLP
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def parse_url_context(url: str) -> str:
    """
    Extracts informative tokens (repo owner, repo name, issue type, issue number)
    from a GitHub URL as a neutral textual fallback.
    """
    if not isinstance(url, str):
        return ""
    m = re.search(r"github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)", url)
    if m:
        owner, repo, kind, num = m.groups()
        repo_clean = repo.replace("-", " ").replace("_", " ")
        return f"{repo_clean} {kind} {num}"
    return ""


def prepare_binary_flakiness_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataset for Task A: Flakiness Detection (Binary Classification: T vs F).
    Guarantees zero data leakage:
    - Excludes reviewer notes (problem_description_excerpt, fix_description_excerpt)
    - Excludes root_cause_category, fix_strategy_category
    - Excludes indeterminate 'ND' and uninspected records
    - Combines issue_title and issue_body into combined_text
    """
    # Filter for valid T / F records
    binary_df = df[df["flakiness_status"].isin(["T", "F"])].copy()

    # Map label to binary integer
    binary_df["label"] = binary_df["flakiness_status"].map({"T": 1, "F": 0})
    binary_df["label_name"] = binary_df["flakiness_status"].map({"T": "Flaky", "F": "False Positive"})

    # Clean text components
    cleaned_titles = binary_df["issue_title"].fillna("").apply(clean_text)
    cleaned_bodies = binary_df["issue_body"].fillna("").apply(clean_text)

    # Fallback for missing title/body using neutral repo / URL tokens
    url_contexts = binary_df["url_issue"].fillna("").apply(parse_url_context)
    repos = binary_df["repository"].fillna("").astype(str).str.replace("-", " ").str.replace("_", " ").str.lower()

    combined_texts = []
    for title, body, u_ctx, repo in zip(cleaned_titles, cleaned_bodies, url_contexts, repos):
        parts = []
        if title:
            parts.append(title)
        elif u_ctx:
            parts.append(u_ctx)
        elif repo:
            parts.append(repo)

        if body:
            parts.append(body)

        combined = " ".join(parts).strip()
        if not combined:
            combined = f"issue in {repo}" if repo else "test failure issue"
        combined_texts.append(combined)

    binary_df["combined_text"] = combined_texts
    binary_df["text_length"] = binary_df["combined_text"].apply(lambda x: len(x.split()))

    # Select model-ready columns
    selected_cols = [
        "issue_id",
        "repository",
        "language",
        "url_issue",
        "flakiness_status",
        "label",
        "label_name",
        "combined_text",
        "text_length"
    ]
    return binary_df[selected_cols]


def prepare_root_cause_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataset for Task B: Root-Cause Classification.
    Uses confirmed flaky issues ('T') with discrete, verified root-cause annotations.
    Excludes '--', 'Hard to Classify', and unannotated records.
    Combines issue report text with verified problem description excerpts.
    """
    excluded_categories = ["--", "-", "nan", "Hard to Classify", "Hard to classify"]

    # Filter for confirmed flaky issues with specific root cause
    rc_df = df[
        (df["flakiness_status"] == "T") &
        (df["root_cause_category"].notna()) &
        (~df["root_cause_category"].isin(excluded_categories))
    ].copy()

    # Standardize category strings
    rc_df["root_cause_category"] = rc_df["root_cause_category"].str.strip()
    category_cleanup = {
        "Hard to classify": "Hard to Classify",
        "*Platform Dependency": "Platform Dependency",
        "*Test Suite Timeout": "Test Suite Timeout",
        "*Test Case Timeout": "Test Case Timeout",
        "*Too Restrictive Range": "Too Restrictive Range"
    }
    rc_df["root_cause_category"] = rc_df["root_cause_category"].replace(category_cleanup)

    # Build comprehensive text for root cause analysis
    cleaned_titles = rc_df["issue_title"].fillna("").apply(clean_text)
    cleaned_bodies = rc_df["issue_body"].fillna("").apply(clean_text)
    cleaned_excerpts = rc_df["problem_description_excerpt"].fillna("").apply(clean_text)
    url_contexts = rc_df["url_issue"].fillna("").apply(parse_url_context)

    combined_texts = []
    for title, body, excerpt, u_ctx in zip(cleaned_titles, cleaned_bodies, cleaned_excerpts, url_contexts):
        parts = []
        if title:
            parts.append(title)
        elif u_ctx:
            parts.append(u_ctx)

        if excerpt:
            parts.append(excerpt)
        if body:
            parts.append(body)

        combined = " ".join(parts).strip()
        combined_texts.append(combined)

    rc_df["combined_text"] = combined_texts
    rc_df["text_length"] = rc_df["combined_text"].apply(lambda x: len(x.split()))

    selected_cols = [
        "issue_id",
        "repository",
        "language",
        "url_issue",
        "root_cause_category",
        "combined_text",
        "text_length"
    ]
    return rc_df[selected_cols]


def run_preprocessing_pipeline(
    raw_path: str = RAW_CSV_PATH,
    output_dir: str = PROCESSED_DIR,
    metrics_dir: str = METRICS_DIR
) -> dict:
    """
    Runs complete preprocessing workflow and writes output CSV files and summary metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"[Preprocessing] Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    initial_count = len(df)

    # 1. Deduplication
    # Remove explicitly flagged duplicate rows and duplicate issue URLs
    df_dedup = df[df["flakiness_status"] != "DUPLICATE"].copy()
    df_dedup = df_dedup.drop_duplicates(subset=["url_issue"]).copy()
    dedup_count = len(df_dedup)
    removed_duplicates = initial_count - dedup_count
    print(f"[Preprocessing] Removed {removed_duplicates} duplicates (remaining: {dedup_count})")

    # 2. Binary Flakiness Dataset (Task A)
    binary_df = prepare_binary_flakiness_dataset(df_dedup)
    binary_path = os.path.join(output_dir, "flaky_binary_dataset.csv")
    binary_df.to_csv(binary_path, index=False)
    print(f"[Preprocessing] Task A Binary dataset saved: {binary_df.shape} -> {binary_path}")

    # 3. Root-Cause Dataset (Task B)
    rc_df = prepare_root_cause_dataset(df_dedup)
    rc_path = os.path.join(output_dir, "root_cause_dataset.csv")
    rc_df.to_csv(rc_path, index=False)
    print(f"[Preprocessing] Task B Root-Cause dataset saved: {rc_df.shape} -> {rc_path}")

    # 4. Full Cleaned Dataset for Explorer
    full_cleaned_path = os.path.join(output_dir, "full_cleaned_dataset.csv")
    df_dedup.to_csv(full_cleaned_path, index=False)
    print(f"[Preprocessing] Full cleaned dataset saved: {df_dedup.shape} -> {full_cleaned_path}")

    summary = {
        "raw_records": initial_count,
        "deduplicated_records": dedup_count,
        "task_a_binary": {
            "total_samples": int(len(binary_df)),
            "flaky_positive (T)": int((binary_df["label"] == 1).sum()),
            "false_positive (F)": int((binary_df["label"] == 0).sum()),
            "imbalance_ratio": float(round((binary_df["label"] == 1).sum() / (binary_df["label"] == 0).sum(), 3)),
            "languages": binary_df["language"].value_counts().to_dict(),
            "avg_text_length_words": float(round(binary_df["text_length"].mean(), 1))
        },
        "task_b_root_cause": {
            "total_samples": int(len(rc_df)),
            "num_classes": int(rc_df["root_cause_category"].nunique()),
            "class_distribution": rc_df["root_cause_category"].value_counts().to_dict(),
            "avg_text_length_words": float(round(rc_df["text_length"].mean(), 1))
        }
    }

    summary_path = os.path.join(metrics_dir, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Preprocessing] Summary written to {summary_path}")

    return summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
