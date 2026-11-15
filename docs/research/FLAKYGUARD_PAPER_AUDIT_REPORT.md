# FlakyGuard AI: Comprehensive Paper Audit & Research Integrity Report

**Date**: November 2, 2026  
**Auditor**: Senior Research Assistant & Empirical Software Engineering Auditor  
**Target Manuscript**: [`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)  
**Accompanying Presentation**: [`docs/research/teacher_presentation_outline.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/teacher_presentation_outline.md)  
**Status**: Integrity Audit Complete & All Discrepancies Reconciled  

---

## 1. Executive Summary of Audit

An exhaustive forensic research audit was performed on the FlakyGuard AI research manuscript. The primary objective was to ensure that every numerical claim, baseline comparison, methodological description, and citation is fully substantiated by reproducible experimental evidence from the project codebase.

### Overall Audit Verdict:
**SCIENTIFICALLY CONSISTENT, METHODOLOGICALLY DEFENSIBLE, AND READY FOR FACULTY REVIEW.**

The revised manuscript eliminates evaluation artifacts, clarifies model-specific uncertainty adjustments, separates accuracy baselines from Macro-F1 baselines, corrects terminology regarding repeated grouped partitions, and tones down speculative causal assertions.

---

## 2. Major Issues Identified & Corrections Made

### Issue 1: Methodological Terminology in Repository-Grouped Partitioning
- **The Issue**: Earlier sections referred to repeated repository evaluations as *"10 independent folds"*, *"10-fold cross-validation"*, or *"independent cross-repository folds"*.
- **The Problem**: Conventional $K$-fold cross-validation partitions the dataset into $K$ mutually disjoint subsets where each sample appears in the test set exactly once. In contrast, `GroupShuffleSplit` across 10 random seeds generates **10 repeated random partitions**. While repository overlap is strictly zero *within* each partition, different repetitions may select overlapping test repositories. The repetitions are therefore not statistically independent folds.
- **The Correction**: Replaced all occurrences with **"10 repeated repository-grouped random partitions"**. Explicitly clarified that mean and standard deviation describe performance variation across repeated partitions.

### Issue 2: Invalid Root-Cause Baseline Comparisons (Accuracy vs. Macro-F1 Conflation)
- **The Issue**: The manuscript previously divided root-cause Macro-F1 scores (0.2834 and 0.4222) by uniform chance accuracy baselines ($1/13 \approx 0.0769$ and $1/4 = 0.2500$) and presented them as "Macro-F1 baselines".
- **The Problem**: Accuracy and Macro-F1 are distinct mathematical metrics. In an imbalanced multi-class problem, the expected uniform-random Macro-F1 is **not** $1/K$. Presenting $1/K$ as a Macro-F1 baseline compares apples to oranges.
- **The Correction**: Implemented [`src/root_cause_baselines.py`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/src/root_cause_baselines.py) to calculate exact empirical baselines for both the 13-class and 4-class settings across 1,000 Monte Carlo trials on the actual test set ($N=118$). Saved to [`artifacts/metrics/root_cause_baselines.csv`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/metrics/root_cause_baselines.csv).
  - *13-Class*: Uniform Random Macro-F1 = **0.0655** (Accuracy: **7.76%**); Majority Macro-F1 = **0.0232** (Accuracy: **17.80%**).
  - *4-Class*: Uniform Random Macro-F1 = **0.2383** (Accuracy: **25.07%**); Majority Macro-F1 = **0.1358** (Accuracy: **37.29%**).
  - Regenerated Figure 8 as a two-panel figure clearly separating Macro-F1 vs. Macro-F1 baselines from Accuracy vs. Accuracy baselines.

### Issue 3: Inaccurate Generalization of Bootstrap Uncertainty Under Clustering
- **The Issue**: Previous text claimed that naive IID bootstrapping *"always underestimates uncertainty by 30–40%"*.
- **The Problem**: Inspection of the actual cluster bootstrap results in [`artifacts/metrics/confidence_interval_report.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/metrics/confidence_interval_report.md) showed:
  - Random Forest: Cluster SE = 0.0349 vs. IID SE = 0.0310 (Ratio: **$1.13\times$**, or +13%)
  - Linear SVM: Cluster SE = 0.0355 vs. IID SE = 0.0341 (Ratio: **$1.04\times$**, or +4%)
  - Logistic Regression: Cluster SE = 0.0303 vs. IID SE = 0.0308 (Ratio: **$0.98\times$**, or -2%)
- **The Correction**: Corrected the text to state that the effect of repository clustering on estimated standard error is **model-specific**. Naive IID bootstrapping does not universally underestimate uncertainty across all models, though it does underestimate standard error for Random Forest.

### Issue 4: Explainability Claim Overreach & Tree MDI Clarification
- **The Issue**: Earlier versions claimed "100% sign invariance across distribution shift" and implied that Random Forest features have stable signs.
- **The Problem**:
  1. In the evaluated 10-token test on Logistic Regression, `goroutine` inverted sign from $-0.1708$ to $+0.0387$, yielding a consistency rate of **90% (9/10), not 100%**.
  2. Random Forest feature importance is based on Mean Decrease in Impurity (Gini importance), which is **mathematically non-negative ($\ge 0$)** and contains no directional sign.
- **The Correction**: Clearly separated Logistic Regression signed coefficients from Random Forest MDI rankings. Clarified that signed instance-level attributions for tree ensembles require game-theoretic Shapley values (SHAP TreeExplainer) or local surrogates (LIME).

### Issue 5: Unsupported Causal Statements in Cross-Language Generalization
- **The Issue**: The manuscript previously asserted definitively that Java performed poorly due to "Spring and JUnit annotation barriers" and that Linear SVM collapsed due to "decision threshold drift".
- **The Problem**: While plausible hypotheses, these mechanisms were not experimentally isolated or proven via controlled ablation studies.
- **The Correction**: Weakened the assertions to careful scientific phrasing: *"An observed possibility is..."*, *"This may reflect..."*, *"However, the present experiment does not isolate this mechanism."*

### Issue 6: Citation Discrepancies and Incomplete Bibliographic Entries
- **The Issue**: Section 7 cited King et al. (2018) and Haben et al. (2021), but neither appeared in the References list. Conversely, Gyori et al. (2015) and Zhang et al. (2014) appeared in the References list without text citations.
- **The Correction**: Added complete, verified bibliographic entries for King et al. (2018) and Haben et al. (2021). Integrated in-text citations for Gyori et al. (2015) and Zhang et al. (2014) into Section 9 (Related Work). Verified that every in-text citation has a reference entry, and every reference is cited.

---

## 3. Numerical Evidence Map & Reconciled Figures

Every numerical statement in the revised manuscript is verified against its source artifact:

| Metric / Scope | Manuscript Value | Source Artifact | Verification Status |
| :--- | :---: | :--- | :---: |
| **Raw Excel Issues / Repositories** | 1,543 / 850 | `data/raw/DataBase_GitHub.xlsx` | **Verified** |
| **Deduplicated Issues / Repositories** | 1,513 / 842 | `data/processed/full_cleaned_dataset.csv` | **Verified** |
| **Binary Dataset Issues / Repositories** | 1,101 / 641 | `data/processed/flaky_binary_dataset.csv` | **Verified** |
| **Root-Cause Dataset Issues / Repositories** | 589 / 349 | `data/processed/root_cause_dataset.csv` | **Verified** |
| **Random Holdout Macro-F1 (RF / LR / SVM)** | 0.6848 / 0.6555 / 0.6107 | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **Random Holdout ROC-AUC (RF / LR / SVM)** | 0.7310 / 0.7262 / 0.6993 | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **Repeated Repo Mean Macro-F1 (LR / RF / SVM)** | 0.6670 / 0.6542 / 0.5201 | `artifacts/metrics/repeated_repository_generalization.csv` | **Verified** |
| **Repeated Repo Std Macro-F1 (LR / RF / SVM)** | 0.0239 / 0.0267 / 0.0546 | `artifacts/metrics/repeated_repository_generalization.csv` | **Verified** |
| **Cluster vs IID Bootstrap SE (RF)** | 0.0349 vs 0.0310 ($1.13\times$) | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **Cluster vs IID Bootstrap SE (LR)** | 0.0303 vs 0.0308 ($0.98\times$) | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **Cluster vs IID Bootstrap SE (SVM)** | 0.0355 vs 0.0341 ($1.04\times$) | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **LOLO Mean Macro-F1 (LR / RF / SVM)** | 0.6053 / 0.6002 / 0.4822 | `artifacts/metrics/lolo_model_selection_audit.md` | **Verified** |
| **LOLO Pooled Macro-F1 (RF / LR / SVM)** | 0.6588 / 0.6539 / 0.5024 | `artifacts/metrics/confidence_interval_report.md` | **Verified** |
| **13-Class RC RF Macro-F1 / Accuracy** | 0.2834 / 38.14% | `artifacts/metrics/root_cause_generalization_report.md` | **Verified** |
| **13-Class Uniform Random Baseline (Macro-F1 / Acc)** | 0.0655 / 7.76% | `artifacts/metrics/root_cause_baselines.csv` | **Verified** |
| **13-Class Majority Baseline (Macro-F1 / Acc)** | 0.0232 / 17.80% | `artifacts/metrics/root_cause_baselines.csv` | **Verified** |
| **4-Class RC LR Macro-F1 / Accuracy** | 0.4222 / 54.24% | `artifacts/metrics/root_cause_generalization_report.md` | **Verified** |
| **4-Class Uniform Random Baseline (Macro-F1 / Acc)** | 0.2383 / 25.07% | `artifacts/metrics/root_cause_baselines.csv` | **Verified** |
| **4-Class Majority Baseline (Macro-F1 / Acc)** | 0.1358 / 37.29% | `artifacts/metrics/root_cause_baselines.csv` | **Verified** |
| **Curated Sign Stability / Inversion** | 9/10 (90%) / `goroutine` | `artifacts/metrics/explainability_claim_audit.md` | **Verified** |

---

## 4. Claims Removed or Weakened

1. **Removed**: Claims that repeated `GroupShuffleSplit` evaluations represent "10 independent folds of cross-validation".
2. **Removed**: Claims that 13-to-4 category aggregation represents a "mathematically proven algorithmic improvement".
3. **Removed**: Comparisons dividing Macro-F1 scores by uniform accuracy ($1/K$).
4. **Weakened**: The claim that naive IID bootstrapping "always underestimates uncertainty by 30–40%" was tempered to model-specific adjustments (+13% for RF, +4% for SVM, -2% for LR).
5. **Weakened**: Definitive statements attributing Java transfer degradation to Spring/JUnit syntax were softened to unisolated hypotheses for future research.

---

## 5. Automated Tests and Verification

All 25 automated unit and regression tests in `tests/` pass with zero failures:
- `tests/test_app_imports.py` (1 test)
- `tests/test_dataset_loading.py` (3 tests)
- `tests/test_inference.py` (3 tests)
- `tests/test_models.py` (3 tests)
- `tests/test_preprocessing.py` (4 tests)
- `tests/test_second_audit.py` (6 tests)
- `tests/test_validation_generalization.py` (5 tests)

**Outcome**: **25 passed in 1.65s** (`pytest tests/`).

---

## 6. Remaining Limitations for Supervisor Discussion

1. **Observational Issue Text**: The system triages developer natural language reports, not dynamic execution traces.
2. **Lexical Representation**: TF-IDF models word frequencies and bigrams, but does not capture deep sequential dependencies or embedded code syntax.
3. **Extreme Minority Scarcity**: Sample counts for rare root causes (e.g. `Test Suite Timeout`, $N=5$) inherently limit supervised learning confidence.
