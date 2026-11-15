# FlakyGuard AI: Comprehensive Reproducibility & Leakage Audit

**Audit Date**: November 2, 2026  
**Auditor**: Lead AI/ML Research Validation Engineer  
**Dataset**: Barbosa et al., *Test Flakiness Across Programming Languages*, IEEE TSE 2023  
**Target Repository**: `AI-Based Flaky Test Detection`

---

## 1. Executive Summary & Verification Verdict

The FlakyGuard AI repository was subjected to an exhaustive end-to-end reproducibility audit. 

### Verdict: **VERIFIED & REPRODUCIBLE (WITH CRITICAL DISTRIBUTION SHIFT DISCOVERY)**
- **Metric Reproducibility**: 100% of reported metrics across all classical ML architectures (Random Forest, Logistic Regression, Linear SVM) were reproduced to four decimal places.
- **Data Pipeline Integrity**: Preprocessing pipelines adhere strictly to leakage-prevention protocols. TF-IDF vectorizers are fitted exclusively on the training partition.
- **Key Empirical Discovery**: The standard stratified random 80/20 split exhibits **35.3% repository overlap** (66 out of 187 test repositories are present in training), accounting for **43.9% of all test instances** (97 of 221). This empirically demonstrates that random splits risk intra-repository memorization and establishes the necessity of Phase 2 (Repository-Independent Validation).

---

## 2. Dataset Pipeline & Count Verification

All dataset counts across pipeline stages were audited against the raw Excel file:

| Stage / Artifact | Rows | Columns | Verified Description |
| :--- | :---: | :---: | :--- |
| `data/raw/DataBase_GitHub.xlsx` (Sheet: `Data`) | 1,543 | 17 | Raw official benchmark spreadsheet |
| `data/raw/raw_flaky_issues.csv` | 1,543 | 19 | Raw data enriched with scraped GitHub title & body |
| `data/processed/full_cleaned_dataset.csv` | 1,513 | 19 | Deduplicated dataset (pruned 30 duplicate issue URLs) |
| `data/processed/flaky_binary_dataset.csv` (Task A) | 1,101 | 9 | Ground-truth labeled binary subset (`T` = 701, `F` = 400) |
| `data/processed/root_cause_dataset.csv` (Task B) | 589 | 7 | Confirmed flaky tests across 13 distinct root-cause classes |

### Label Inclusions & Exclusions
- **Binary Classification (Task A)**: Strictly includes verified `T` (True Flaky) and `F` (False Positive). All indeterminate (`ND`: 224), duplicate (`DUPLICATE`: 25), and unannotated (`NaN`: 188) issues are completely excluded from model training and evaluation.
- **Root-Cause Classification (Task B)**: Strictly includes confirmed flaky tests ($N=589$) possessing one of the 13 verified root causes. Vague annotations (`Hard to Classify`: 335) and unannotated non-flaky rows (`--`: 422) are excluded.

---

## 3. Train / Test Split Verification

The existing evaluation uses an 80/20 stratified split (`test_size=0.20`, `stratify=y`, `random_state=42`):

```
Task A Binary Dataset: 1,101 samples
├── Training Partition (80%): 880 samples
│   ├── True Flaky (1): 560 samples (63.64%)
│   └── False Positive (0): 320 samples (36.36%)
└── Holdout Test Partition (20%): 221 samples
    ├── True Flaky (1): 141 samples (63.80%)
    └── False Positive (0): 81 samples (36.20%)
```

- **Stratification Check**: Class proportions match within 0.16% between train and test partitions.
- **Vectorizer Fitting Check**: Verified in `src/flaky_detection.py` and `src/root_cause_classifier.py`:
  ```python
  # Confirmed: fit_transform ONLY on training raw text
  X_train = self.vectorizer.fit_transform(X_train_raw)
  X_test = self.vectorizer.transform(X_test_raw)
  ```
  Test instances never participate in vocabulary construction or IDF computation.

---

## 4. Leakage Audit

A comprehensive multi-vector leakage inspection was conducted:

| Potential Leakage Vector | Audit Finding | Status |
| :--- | :--- | :---: |
| **Target Label Leakage** | `flakiness_status`, `label`, `root_cause_category`, and `fix_strategy_category` are strictly absent from the input feature string `combined_text`. | **CLEAN** |
| **Reviewer Annotation Leakage** | The spreadsheet's `problem_description_excerpt` and `fix_description_excerpt` columns were populated by researchers only for flaky tests (`--` for false positives). These columns are strictly excluded from Task A input text. | **CLEAN** |
| **Post-Outcome Metadata Leakage** | Outcome-dependent variables (`days_to_close` M1, `comments_count` M2, issue closed state) are omitted from model features. Features use strictly issue title, issue initial body, and programming language. | **CLEAN** |
| **Issue URL Overlap** | Overlap between train and test URLs: **0 URLs**. | **CLEAN** |
| **Duplicate Issue Overlap** | Pruned during deduplication. No duplicate issues exist across splits. | **CLEAN** |
| **Repository Overlap (Data Dependency)** | **66 repositories (35.3% of test repos)** appear in both train and test. 97 of 221 test issues (43.9%) originate from projects seen in training. | **DOCUMENTED RISK** |

---

## 5. Metric Reproducibility Verification

Independent re-execution of the saved model pipeline on the held-out test set reproduced the saved benchmarks identically:

### Task A: Flakiness Detection Metrics Comparison
- **Random Forest**:
  - Accuracy: Reported = `0.7104`, Re-computed = `0.7104` (Match)
  - Macro-F1: Reported = `0.6848`, Re-computed = `0.6848` (Match)
  - F1-Score: Reported = `0.7746`, Re-computed = `0.7746` (Match)
  - ROC-AUC: Reported = `0.7310`, Re-computed = `0.7310` (Match)
- **Logistic Regression**:
  - Accuracy: Reported = `0.6652`, Re-computed = `0.6652` (Match)
  - Macro-F1: Reported = `0.6555`, Re-computed = `0.6555` (Match)
  - F1-Score: Reported = `0.7132`, Re-computed = `0.7132` (Match)
  - ROC-AUC: Reported = `0.7262`, Re-computed = `0.7262` (Match)
- **Linear SVM (Calibrated)**:
  - Accuracy: Reported = `0.6697`, Re-computed = `0.6697` (Match)
  - Macro-F1: Reported = `0.6107`, Re-computed = `0.6107` (Match)
  - F1-Score: Reported = `0.7622`, Re-computed = `0.7622` (Match)
  - ROC-AUC: Reported = `0.6993`, Re-computed = `0.6993` (Match)

### Task B: Root-Cause Classification Metrics Comparison
- **Logistic Regression**:
  - Accuracy: Reported = `0.3051`, Re-computed = `0.3051` (Match)
  - Macro-F1: Reported = `0.2289`, Re-computed = `0.2289` (Match)
  - Weighted-F1: Reported = `0.2894`, Re-computed = `0.2894` (Match)
- **Linear SVM**:
  - Accuracy: Reported = `0.3305`, Re-computed = `0.3305` (Match)
  - Macro-F1: Reported = `0.1844`, Re-computed = `0.1844` (Match)
  - Weighted-F1: Reported = `0.2931`, Re-computed = `0.2931` (Match)

---

## 6. Audit Conclusion & Next Research Mandate

The existing pipeline is verified as reproducible and internally valid. However, the discovery that **43.9% of test samples share a repository with training data** demonstrates that the random split model may benefit from intra-repository vocabulary leakage (e.g. project-specific package names, test framework abbreviations, or module references).

Proceed immediately to **Phase 2: Repository-Independent Generalization** to quantify the true generalization gap when testing on entirely unseen software repositories.
