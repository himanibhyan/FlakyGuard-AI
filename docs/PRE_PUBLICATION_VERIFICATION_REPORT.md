# FlakyGuard AI: Pre-Publication Verification Report

**Date**: November 15, 2026  
**Auditor**: Independent Senior Machine Learning Researcher, Empirical Software Engineering Auditor, and Release Specialist  
**Repository Working Directory**: `/Users/himani/Downloads/AI-Based Flaky Test Detection`  
**Current Git Commit**: `8b6676d` (`HEAD -> main`)  

---

## 1. Executive Summary

This report establishes the final empirical and technical verification of the **FlakyGuard AI** research framework prior to public release on GitHub. 

The audit resolved:
1. The confusion matrix and true positive accounting discrepancy in RQ5.
2. Complete Git history privacy verification following the initial commit amendment.
3. Complete 8-step dataset reconciliation tracing all 1,543 raw records to the final modeling partitions with 0 residual.
4. Definitive licensing and dataset provenance verification for the underlying TSE22 benchmark (CC0-1.0 Public Domain Dedication).
5. Comprehensive manuscript and documentation cross-check for all canonical metrics.

---

## 2. Issue 1: RQ5 13-Class Confusion Matrix & Metric Accounting

### 2.1 Problem Statement
An earlier report draft cited Accuracy = $44/118 = 37.29\%$, while a per-class list contained a typographical discrepancy. 

### 2.2 Source Artifacts Inspected
- `data/processed/root_cause_dataset.csv` ($N=589$ confirmed flaky issues)
- `src/root_cause_validation.py` (canonical multi-class training and holdout evaluation script)
- `artifacts/metrics/rq5_canonical_predictions.csv` (newly generated test prediction array)
- `artifacts/metrics/rq5_canonical_per_class_metrics.csv` (newly generated exact per-class metrics)

### 2.3 Exact Recalculation from Test Predictions ($N=118$)
Evaluating `RandomForestClassifier(n_estimators=150, max_depth=25, class_weight='balanced', random_state=42)` with TF-IDF (`max_features=8000, min_df=2`) on the stratified 20% holdout partition:
- **Total Test Instances**: $N=118$
- **Total Correct Predictions (Sum of Diagonal)**: **44**
- **Overall Accuracy**: $44 / 118 = \mathbf{0.3729}$ ($37.2881\%$)
- **Macro-F1**: $\mathbf{0.2834}$ ($0.283436$)
- **Weighted-F1**: $\mathbf{0.3552}$ ($0.355154$)
- **Macro Precision**: $\mathbf{0.3328}$ ($0.332846$)
- **Macro Recall**: $\mathbf{0.3023}$ ($0.302267$)

### 2.4 Complete 13-Class Breakdown Table

| Index | Category Name | Test Support | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Precision | Recall | F1-Score | Status & Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | **Async Wait** | 21 | **8** | 13 | 15 | 0.3478 | 0.3810 | 0.3636 | Confused with Timeout & Platform |
| 1 | **Concurrency** | 19 | **5** | 14 | 2 | 0.7143 | 0.2632 | 0.3846 | High precision, conservative recall |
| 2 | **Floating Point Operations**| 3 | **0** | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | Sample scarcity ($N=3$) |
| 3 | **Network** | 14 | **3** | 11 | 7 | 0.3000 | 0.2143 | 0.2500 | Overlap with Resource Leak |
| 4 | **Platform Dependency** | 17 | **9** | 8 | 20 | 0.3103 | 0.5294 | 0.3913 | Moderate recall, broad FP catch |
| 5 | **Randomness** | 4 | **1** | 3 | 0 | 1.0000 | 0.2500 | 0.4000 | Perfect precision (1.0000) |
| 6 | **Resource Leak** | 12 | **8** | 4 | 7 | 0.5333 | 0.6667 | 0.5926 | Strong discriminative vocabulary |
| 7 | **Test Case Timeout** | 6 | **4** | 2 | 4 | 0.5000 | 0.6667 | 0.5714 | Good discriminative vocabulary |
| 8 | **Test Order Dependency** | 8 | **1** | 7 | 5 | 0.1667 | 0.1250 | 0.1429 | Low recall |
| 9 | **Test Suite Timeout** | 1 | **0** | 1 | 4 | 0.0000 | 0.0000 | 0.0000 | Extreme tail scarcity ($N=1$) |
| 10 | **Time** | 6 | **0** | 6 | 1 | 0.0000 | 0.0000 | 0.0000 | Confused with Async Wait |
| 11 | **Too Restrictive Range** | 1 | **0** | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | Extreme tail scarcity ($N=1$) |
| 12 | **Unordered Collections** | 6 | **5** | 1 | 6 | 0.4545 | 0.8333 | 0.5882 | Strong recall (5/6 detected) |
| — | **Total / Macro Average** | **118** | **44** | **74** | **74** | **0.3328** | **0.3023** | **0.2834** | **Accuracy = 44/118 (37.29%)** |

$$\text{Sum of True Positives} = 8 + 5 + 0 + 3 + 9 + 1 + 8 + 4 + 1 + 0 + 0 + 0 + 5 = \mathbf{44}$$

### 2.5 Full 13 × 13 Confusion Matrix
$$\begin{pmatrix}
8 & 2 & 0 & 3 & 4 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 2 \\
2 & 5 & 1 & 1 & 3 & 0 & 2 & 0 & 3 & 2 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 2 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
3 & 0 & 0 & 3 & 3 & 0 & 0 & 3 & 0 & 1 & 1 & 0 & 0 \\
2 & 0 & 1 & 0 & 9 & 0 & 3 & 0 & 1 & 0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 & 2 & 0 & 8 & 0 & 0 & 0 & 0 & 1 & 0 \\
1 & 0 & 0 & 1 & 0 & 0 & 0 & 4 & 0 & 0 & 0 & 0 & 0 \\
2 & 0 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 2 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 & 3 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 5
\end{pmatrix}$$

---

## 3. Issue 2: Git History Privacy Verification

### 3.1 Operations Executed
1. In-place redaction of developer email patterns to `[REDACTED_EMAIL]` in:
   - `data/raw/raw_flaky_issues.csv`
   - `data/raw/github_issues_metadata.json`
   - `data/processed/full_cleaned_dataset.csv`
2. Preserved local unredacted research data in `data/raw/.local_research_backup/`.
3. Added `data/raw/.local_research_backup/` and `data/raw/*_unredacted*` to `.gitignore`.
4. Staged all clean files and executed:
   ```bash
   git commit --amend -m "feat: complete reproducible FlakyGuard AI research framework and validation suite"
   git reflog expire --expire=now --all
   git gc --prune=now
   ```

### 3.2 Verification Results Across All Git Scopes

| Audit Check | Command Executed | Result | Verdict |
| :--- | :--- | :--- | :---: |
| **Active Branches** | `git branch -a` | `* main` | **Clean** |
| **Git Tags** | `git tag -l` | None (0 tags) | **Clean** |
| **Reachable Commits** | `git rev-list --all` | Single commit: `8b6676d9b8222cc434099a239676f26b18f9b45d` | **Clean** |
| **Tracked Files in HEAD** | `git ls-files` | 161 tracked files | **Clean** |
| **Untracked Backup Status**| `git check-ignore data/raw/.local_research_backup/` | `data/raw/.local_research_backup/` (Ignored) | **Clean** |
| **Backup Files Tracked?** | `git ls-files data/raw/.local_research_backup/` | None (0 files tracked) | **Clean** |
| **Email Regex on Tracked Files**| Scanned all 161 files in `HEAD` using `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` | 0 unredacted email addresses | **Clean** |
| **Working Tree Status** | `git status` | Clean working tree (`nothing to commit, working tree clean`) | **Clean** |

---

## 4. Issue 3: Complete Dataset Reconciliation

The table below establishes the complete, 100% accounted-for record provenance from the raw Excel spreadsheet to the modeling datasets:

```
[1,543 Raw Records (DataBase_GitHub.xlsx, sheet: Data)]
  │
  ├── 25 marked flakiness_status == "DUPLICATE"
  │     └── Excluded in Stage 1 Deduplication (leaving 1,518)
  │
  ├── 5 redundant URL Issue rows
  │     ├── 2 Flaky (T) duplicates
  │     ├── 1 False Positive (F) duplicate
  │     └── 2 Not Determined (ND) duplicates
  │     └── Excluded in Stage 2 URL Deduplication (leaving 1,513)
  │
  └── [1,513 Deduplicated Issues]
        │
        ├── 701 Flaky (T) ──────────────────────────┐
        │                                           │
        ├── 400 False Positive (F) ─────────────────┼──> [1,101 Binary Classification Dataset]
        │                                           │    (Task A: 701 Flaky T, 400 Non-Flaky F)
        ├── 222 Not Determined (ND) ──> Excluded    │    (641 Canonical Repositories)
        │   (Lacks confirmed label)                 │
        └── 190 Unannotated (NaN)   ──> Excluded    │
            (Missing reviewer label)                │
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
   [701 Confirmed Flaky Issues (T)]
        │
        ├── 112 Excluded from Root-Cause Classification:
        │     ├── 111 labeled "Hard to Classify"
        │     └── 1 labeled "Hard to classify"
        │
        └── [589 Task B Root-Cause Classification Dataset]
              (13 Discrete Categories, 349 Canonical Repositories)
```

### Complete Filtering Breakdown Table

| Stage | Starting Count | Records Removed | Output Count | Category Breakdown of Output | Reason for Exclusion / Retention |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **1. Raw Excel Records** | — | — | **1,543** | `T`: 703, `F`: 401, `ND`: 224, `NaN`: 190, `DUPLICATE`: 25 | All rows in sheet `Data` (header on row 2) |
| **2. Status Deduplication** | 1,543 | 25 | **1,518** | `T`: 703, `F`: 401, `ND`: 224, `NaN`: 190 | Pruned 25 explicitly flagged `DUPLICATE` issues |
| **3. URL Deduplication** | 1,518 | 5 | **1,513** | `T`: 701, `F`: 400, `ND`: 222, `NaN`: 190 | Pruned 5 redundant issue URLs (2 `T`, 1 `F`, 2 `ND`) |
| **4. Binary Eligibility** | 1,513 | 412 | **1,101** | `T` (Flaky): 701 (63.67%), `F` (False Positive): 400 (36.33%) | Excluded 222 indeterminate `ND` + 190 unlabeled `NaN` |
| **5. Task A Binary Dataset** | 1,101 | 0 | **1,101** | Label 1 (Flaky): 701, Label 0 (False Positive): 400 | Final binary dataset across 641 canonical repos |
| **6. Root-Cause Eligibility** | 701 | 112 | **589** | 13 discrete empirical root-cause classes | Excluded 112 uninformative (`Hard to Classify`) records |
| **7. Task B Dataset** | 589 | 0 | **589** | Async Wait (103), Concurrency (97), Platform (86)... | Final root-cause dataset across 349 canonical repos |

**Mathematical Verification**:
- Stage 1: $703 + 401 + 224 + 190 + 25 = 1,543$ (Residual: **0**)
- Stage 3: $701 + 400 + 222 + 190 = 1,513$ (Residual: **0**)
- Stage 4: $1,101 + 412 = 1,513$ (Residual: **0**)
- Stage 6: $589 + 112 = 701$ (Residual: **0**)

---

## 5. Issue 4: Licensing and Dataset Provenance Verification

### 5.1 Investigation of the Original Benchmark Package
- **Source Repository**: `https://github.com/Test-Flaky/TSE22`
- **Associated Publication**: Barbosa et al., *"Test Flakiness Across Programming Languages"*, *IEEE Transactions on Software Engineering (TSE)*, 2023.
- **Repository License**: **Creative Commons Zero v1.0 Universal (CC0-1.0)**.

### 5.2 Legal Scope of CC0-1.0
Under the terms of CC0-1.0 Universal:
- The authors have dedicated the work to the public domain by waiving all rights worldwide under copyright law.
- The dedication explicitly covers the dataset files (`data/DataBase_GitHub.xlsx`), scripts, and metadata.
- Third parties may copy, modify, distribute, and perform the work without asking permission or negotiating licensing fees.

### 5.3 Attribution & Licensing Separation
- **Third-Party Data**: Covered by CC0-1.0 (Public Domain). Academic citation of Barbosa et al. (*IEEE TSE 2023*) satisfies academic ethics.
- **FlakyGuard AI Code & Artifacts**: Covered by the newly created **MIT License** in [`LICENSE`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/LICENSE).
- **Redistribution Permission**: Fully established. The raw Excel file `data/raw/DataBase_GitHub.xlsx` may safely be distributed in the repository.

---

## 6. Issue 5: Final Manuscript and Repository Consistency

Every occurrence of experimental metrics across all project documents was checked:
- **Manuscript** ([`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)):
  - RQ3 LOLO sample standard deviations: $0.6053 \pm 0.0785$ (LR), $0.6002 \pm 0.0729$ (RF), $0.4822 \pm 0.0621$ (SVM).
  - RQ5 13-Class Random Forest: Macro-F1 = **0.2834**, Accuracy = **37.29%** ($44/118$).
  - RQ5 4-Class Logistic Regression: Macro-F1 = **0.4222**, Accuracy = **48.31%** ($57/118$).
- **README** ([`README.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/README.md)):
  - Reconciled LOLO standard deviations and RQ5 canonical metrics.
- **Model Evaluation Report** ([`docs/MODEL_EVALUATION_REPORT.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/MODEL_EVALUATION_REPORT.md)):
  - Reconciled LOLO standard deviations and RQ5 canonical metrics.
- **Faculty Defense Q&A** ([`docs/FACULTY_DEFENSE_QA.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/FACULTY_DEFENSE_QA.md)):
  - Q21, Q22, and Q29 reconciled to 37.29%, 48.31%, and $\pm 0.0785$.
- **Publication Figures** (`artifacts/figures/publication/`):
  - Figure 7 (Confusion Matrix, $N_{\text{correct}} = 44$) and Figure 8 (13 vs 4-class comparison, Accuracy = 37.29% and 48.31%) regenerated and verified.

---

## 7. Remaining Methodological Limitations

1. **Textual Proxy Scope**: Predictions rely strictly on developer issue descriptions, bug reports, and discussions. The system does not analyze dynamic JVM runtime thread schedules, race detector traces, or CI/CD execution logs.
2. **Minority Class Sparsity**: Tail failure modes (`Test Suite Timeout`: $N=5$, `Too Restrictive Range`: $N=6$) have only 1 holdout test instance each, bounding fine-grained 13-class Macro-F1.
3. **Cross-Language Synthesis**: Cross-language transfer performance reflects shared non-deterministic vocabulary (`intermittent`, `flaky`, `timeout`), but is influenced by community-specific discussion idioms.

---

## 8. Final Three Readiness Verdicts

### Verdict 1: Software Readiness
**Status**: **VERIFIED**  
- All 27 automated unit and integration tests pass cleanly (`27 passed in 2.41s`).
- CLI (`src/inference.py`), training pipelines, and multi-page Streamlit application (`app.py`) execute without errors.

### Verdict 2: Academic Manuscript Readiness
**Status**: **VERIFIED**  
- The manuscript (`FLAKYGUARD_AI_RESEARCH_PAPER.md`) is completely reconciled with authoritative experimental artifacts.
- Baseline lifts, cluster bootstrap confidence intervals, and 15 DOI-verified literature citations are scientifically sound and defensible for supervisor and peer review.

### Verdict 3: Public GitHub Release Readiness
**Status**: **VERIFIED**  
- Git history contains a single amended commit (`8b6676d`) with 0 developer email addresses.
- CC0-1.0 dataset provenance and MIT repository licensing are legally verified.
- The working tree is clean and ready for public publication.
