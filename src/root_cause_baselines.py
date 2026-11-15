"""
Root-Cause Classification Empirical Baselines Generator (Phase 3 Audit).
Calculates true empirical baselines for both 13-class and 4-class root-cause classification:
1. Majority-Class Baseline (Zero-Rule)
2. Uniform-Random Baseline (Monte Carlo B=1000, seed=42)
3. Stratified-Prior Baseline (Monte Carlo B=1000, seed=42)
Reports Accuracy, Macro-F1, and Weighted-F1 on the exact holdout test set (N=118).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics"))


def compute_baselines():
    os.makedirs(METRICS_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "root_cause_dataset.csv"))
    rng = np.random.default_rng(42)

    records = []

    # ---------------------------------------------------------
    # 1. 13-Class Discrete Setting
    # ---------------------------------------------------------
    le13 = LabelEncoder()
    y13 = le13.fit_transform(df["root_cause_category"])
    _, _, y_tr13, y_te13 = train_test_split(
        df["combined_text"], y13, test_size=0.20, stratify=y13, random_state=42
    )
    n_te13 = len(y_te13)

    # A: Majority Class
    maj_13 = np.bincount(y_tr13).argmax()
    maj_name_13 = le13.classes_[maj_13]
    pred_maj_13 = np.full(n_te13, maj_13)
    records.append({
        "setting": "13-Class Discrete",
        "baseline_type": f"Majority Class ({maj_name_13})",
        "accuracy": round(accuracy_score(y_te13, pred_maj_13), 4),
        "macro_f1": round(f1_score(y_te13, pred_maj_13, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_te13, pred_maj_13, average="weighted", zero_division=0), 4),
        "notes": "Always predicts the most frequent class in training set"
    })

    # B: Uniform Random (B=1000)
    acc_u13, mf1_u13, wf1_u13 = [], [], []
    for _ in range(1000):
        p = rng.integers(0, 13, size=n_te13)
        acc_u13.append(accuracy_score(y_te13, p))
        mf1_u13.append(f1_score(y_te13, p, average="macro", zero_division=0))
        wf1_u13.append(f1_score(y_te13, p, average="weighted", zero_division=0))
    records.append({
        "setting": "13-Class Discrete",
        "baseline_type": "Uniform Random (Expected)",
        "accuracy": round(float(np.mean(acc_u13)), 4),
        "macro_f1": round(float(np.mean(mf1_u13)), 4),
        "weighted_f1": round(float(np.mean(wf1_u13)), 4),
        "notes": "Random class drawn with uniform 1/13 probability"
    })

    # C: Stratified Prior (B=1000)
    p_prior_13 = np.bincount(y_tr13) / len(y_tr13)
    acc_s13, mf1_s13, wf1_s13 = [], [], []
    for _ in range(1000):
        p = rng.choice(13, size=n_te13, p=p_prior_13)
        acc_s13.append(accuracy_score(y_te13, p))
        mf1_s13.append(f1_score(y_te13, p, average="macro", zero_division=0))
        wf1_s13.append(f1_score(y_te13, p, average="weighted", zero_division=0))
    records.append({
        "setting": "13-Class Discrete",
        "baseline_type": "Stratified Random Prior (Expected)",
        "accuracy": round(float(np.mean(acc_s13)), 4),
        "macro_f1": round(float(np.mean(mf1_s13)), 4),
        "weighted_f1": round(float(np.mean(wf1_s13)), 4),
        "notes": "Random class drawn according to training class frequencies"
    })

    # ---------------------------------------------------------
    # 2. 4-Class Macroscopic Setting
    # ---------------------------------------------------------
    macro_mapping = {
        "Async Wait": "Timing & Asynchrony", "Time": "Timing & Asynchrony",
        "Test Case Timeout": "Timing & Asynchrony", "Test Suite Timeout": "Timing & Asynchrony",
        "Concurrency": "Concurrency & Non-Determinism", "Randomness": "Concurrency & Non-Determinism",
        "Unordered Collections": "Concurrency & Non-Determinism",
        "Network": "Resource & Environment", "Resource Leak": "Resource & Environment",
        "Platform Dependency": "Resource & Environment",
        "Test Order Dependency": "Test Logic & Assertions",
        "Floating Point Operations": "Test Logic & Assertions",
        "Too Restrictive Range": "Test Logic & Assertions"
    }
    df["macro"] = df["root_cause_category"].map(macro_mapping)
    le4 = LabelEncoder()
    y4 = le4.fit_transform(df["macro"])
    _, _, y_tr4, y_te4 = train_test_split(
        df["combined_text"], y4, test_size=0.20, stratify=y4, random_state=42
    )
    n_te4 = len(y_te4)

    # A: Majority Class
    maj_4 = np.bincount(y_tr4).argmax()
    maj_name_4 = le4.classes_[maj_4]
    pred_maj_4 = np.full(n_te4, maj_4)
    records.append({
        "setting": "4-Class Macroscopic",
        "baseline_type": f"Majority Class ({maj_name_4})",
        "accuracy": round(accuracy_score(y_te4, pred_maj_4), 4),
        "macro_f1": round(f1_score(y_te4, pred_maj_4, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_te4, pred_maj_4, average="weighted", zero_division=0), 4),
        "notes": "Always predicts the most frequent macro class in training set"
    })

    # B: Uniform Random (B=1000)
    acc_u4, mf1_u4, wf1_u4 = [], [], []
    for _ in range(1000):
        p = rng.integers(0, 4, size=n_te4)
        acc_u4.append(accuracy_score(y_te4, p))
        mf1_u4.append(f1_score(y_te4, p, average="macro", zero_division=0))
        wf1_u4.append(f1_score(y_te4, p, average="weighted", zero_division=0))
    records.append({
        "setting": "4-Class Macroscopic",
        "baseline_type": "Uniform Random (Expected)",
        "accuracy": round(float(np.mean(acc_u4)), 4),
        "macro_f1": round(float(np.mean(mf1_u4)), 4),
        "weighted_f1": round(float(np.mean(wf1_u4)), 4),
        "notes": "Random class drawn with uniform 1/4 probability"
    })

    # C: Stratified Prior (B=1000)
    p_prior_4 = np.bincount(y_tr4) / len(y_tr4)
    acc_s4, mf1_s4, wf1_s4 = [], [], []
    for _ in range(1000):
        p = rng.choice(4, size=n_te4, p=p_prior_4)
        acc_s4.append(accuracy_score(y_te4, p))
        mf1_s4.append(f1_score(y_te4, p, average="macro", zero_division=0))
        wf1_s4.append(f1_score(y_te4, p, average="weighted", zero_division=0))
    records.append({
        "setting": "4-Class Macroscopic",
        "baseline_type": "Stratified Random Prior (Expected)",
        "accuracy": round(float(np.mean(acc_s4)), 4),
        "macro_f1": round(float(np.mean(mf1_s4)), 4),
        "weighted_f1": round(float(np.mean(wf1_s4)), 4),
        "notes": "Random class drawn according to training macro class frequencies"
    })

    out_df = pd.DataFrame(records)
    csv_path = os.path.join(METRICS_DIR, "root_cause_baselines.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"[Baselines Audit] Saved empirical baselines to: {csv_path}")
    print(out_df.to_string())
    return out_df


if __name__ == "__main__":
    compute_baselines()
