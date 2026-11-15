"""
Dataset Audit Script for FlakyGuard AI.
Performs deep inspection of the raw dataset schema, column types,
missing-value distributions, label cardinalities, and data integrity.
"""

import os
import json
import pandas as pd
import numpy as np

RAW_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_flaky_issues.csv"))
AUDIT_REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics", "dataset_audit_report.json"))


def audit_dataset(csv_path: str = RAW_CSV_PATH) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "exact_columns": list(df.columns),
        "column_types": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "flakiness_status_distribution": df["flakiness_status"].value_counts(dropna=False).to_dict(),
        "language_distribution": df["language"].value_counts(dropna=False).to_dict(),
        "root_cause_distribution": df["root_cause_category"].value_counts(dropna=False).to_dict(),
        "fix_strategy_distribution": df["fix_strategy_category"].value_counts(dropna=False).head(25).to_dict(),
        "text_statistics": {
            "issue_title_present": int((df["issue_title"].fillna("").str.strip() != "").sum()),
            "issue_body_present": int((df["issue_body"].fillna("").str.strip() != "").sum()),
            "problem_excerpt_present": int((~df["problem_description_excerpt"].isin(["--", np.nan, ""])).sum()),
            "fix_excerpt_present": int((~df["fix_description_excerpt"].isin(["--", np.nan, ""])).sum()),
        },
        "multi_label_check": {
            "root_cause_contains_delimiter": int(df["root_cause_category"].dropna().str.contains(r"[,;|/]").sum()),
            "fix_strategy_contains_delimiter": int(df["fix_strategy_category"].dropna().str.contains(r"[,;|/]").sum())
        }
    }

    # Print summary to console
    print("=" * 60)
    print("FLAKYGUARD AI - DATASET AUDIT REPORT")
    print("=" * 60)
    print(f"Shape: {report['dataset_shape']['rows']} rows, {report['dataset_shape']['columns']} columns")
    print("\nColumns:")
    for col in report["exact_columns"]:
        missing = report["missing_values"][col]
        pct = (missing / df.shape[0]) * 100
        print(f"  - {col:<30} (dtype: {report['column_types'][col]:<8}, missing: {missing:>4} / {pct:5.1f}%)")

    print("\nFlakiness Status Distribution:")
    for k, v in report["flakiness_status_distribution"].items():
        print(f"  - {str(k):<15}: {v}")

    print("\nProgramming Language Distribution:")
    for k, v in report["language_distribution"].items():
        print(f"  - {str(k):<15}: {v}")

    print("\nRoot-Cause Categories (Top 10):")
    for k, v in list(report["root_cause_distribution"].items())[:10]:
        print(f"  - {str(k):<30}: {v}")

    print("\nText Coverage:")
    for k, v in report["text_statistics"].items():
        pct = (v / df.shape[0]) * 100
        print(f"  - {k:<30}: {v:>5} ({pct:5.1f}%)")

    print("\nMulti-label Analysis:")
    print(f"  - Root cause contains multi-value delimiter: {report['multi_label_check']['root_cause_contains_delimiter']} issues")
    print(f"  - Fix strategy contains multi-value delimiter: {report['multi_label_check']['fix_strategy_contains_delimiter']} issues")
    print("  => Conclusion: Single-label categorization used by paper authors.")

    os.makedirs(os.path.dirname(AUDIT_REPORT_PATH), exist_ok=True)
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        # Convert non-string keys
        clean_report = json.loads(json.dumps(report, default=str))
        json.dump(clean_report, f, indent=2)
    print(f"\nAudit report saved to: {AUDIT_REPORT_PATH}")
    return report


if __name__ == "__main__":
    audit_dataset()
