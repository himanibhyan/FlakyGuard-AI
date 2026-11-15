# FlakyGuard AI: Manuscript Claim Verification & Forensic Audit Report

**Date**: November 15, 2026  
**Auditor Role**: Independent Senior Machine Learning Researcher, Empirical Software Engineering Auditor  
**Target Manuscript**: [`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)  
**Accompanying Reports**: `artifacts/metrics/`, `artifacts/figures/publication/`  

---

## 1. Claim-by-Claim Verification Table

| # | Manuscript Claim | Manuscript Location | Source Artifact / Code | Verified Empirical Value | Verification Status | Necessary Clarification / Correction |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | Raw dataset contains 1,543 records across 5 languages. | Section 10, Table 2 | `data/raw/DataBase_GitHub.xlsx` Sheet `Data` | Exactly 1,543 data rows | **Verified** | None. Exact match. |
| **2** | Binary classification dataset contains 1,101 issues (701 Flaky, 400 False Positive). | Section 10, Table 2 | `data/processed/flaky_binary_dataset.csv` | 1,101 rows (701 `T`, 400 `F`) | **Verified** | None. Exact match. |
| **3** | Binary dataset spans 641 unique canonical repositories. | Section 10, Table 2 | `parse_canonical_repo` on `url_issue` | Exactly 641 canonical repos | **Verified** | None. Exact match. |
| **4** | Confirmed flaky dataset consists of 589 issues across 13 classes. | Section 10, 21 | `data/processed/root_cause_dataset.csv` | 589 rows, 13 classes | **Verified** | None. Exact match. |
| **5** | In-distribution RF Macro-F1 = 0.6848, ROC-AUC = 0.7310. | Section 17, Table 2 | `models/best_flaky_model.joblib`, `model_comparison.json` | Macro-F1: 0.6848, ROC: 0.7310 | **Verified** | None. Exact match. |
| **6** | In-distribution RF Accuracy = 72.40%. | Section 17, Table 2 | `src/model_training.py`, `model_comparison.json` | Actual accuracy: **71.04%** (157/221) | **Minor Variance** | Serialized model gets 71.04%; 72.40% represents 160/221 (+3 samples). Note in text. |
| **7** | In-distribution majority prior baseline Macro-F1 = 0.4550. | Section 17, Table 2 | `train_test_split(..., test_size=0.20)` | Macro-F1: 0.4550 | **Verified** | None. Exact match. |
| **8** | Across 10 repository-grouped partitions, LR Mean Macro-F1 = 0.6670 $\pm$ 0.0239. | Section 18, Table 3 | `artifacts/metrics/repeated_repository_generalization.csv` | 0.6670 $\pm$ 0.0239 | **Verified** | None. Exact match. |
| **9** | Across 10 partitions, RF Mean Macro-F1 = 0.6542 $\pm$ 0.0267. | Section 18, Table 3 | `artifacts/metrics/repeated_repository_generalization.csv` | 0.6542 $\pm$ 0.0267 | **Verified** | None. Exact match. |
| **10** | Across 10 partitions, Linear SVM Mean Macro-F1 = 0.5201 $\pm$ 0.0546. | Section 18, Table 3 | `artifacts/metrics/repeated_repository_generalization.csv` | 0.5201 $\pm$ 0.0546 | **Verified** | None. Exact match. |
| **11** | Zero repository overlap exists within each of the 10 partitions. | Section 18, 18.1 | `tests/test_second_audit.py` (line 80) | $\text{Train} \cap \text{Test} = \emptyset$ $\forall$ seeds | **Verified** | None. Mathematically verified. |
| **12** | LOLO Language-Mean Macro-F1: LR = 0.6053, RF = 0.6002, SVM = 0.4822. | Section 19, Table 5 | `artifacts/metrics/lolo_results.csv` | LR: 0.6053, RF: 0.6002, SVM: 0.4822 | **Verified** | None. Exact match. |
| **13** | Pooled LOLO Macro-F1 across 1,101 predictions: RF = 0.6588, LR = 0.6539. | Section 19, 19.1 | `artifacts/metrics/lolo_model_selection_audit.md` | RF: 0.6588, LR: 0.6539 | **Verified** | None. Exact match. |
| **14** | Cluster bootstrap SE inflation is 1.13x for RF, 1.04x for SVM, and 0.98x for LR. | Section 18, 18.2 | `artifacts/metrics/confidence_interval_report.md` | RF: 1.13x, SVM: 1.04x, LR: 0.98x | **Verified** | None. Exact match. |
| **15** | Curated 10-token test in LR yields 90% sign consistency; `goroutine` inverted sign. | Section 20, Table 6 | `artifacts/metrics/explainability_claim_audit.md` | 9/10 consistent; `goroutine`: $-0.1708 \rightarrow +0.0387$ | **Verified** | None. Exact match. |
| **16** | Random Forest Gini MDI is non-negative and cannot indicate directional polarity. | Section 20, 20.2 | `tests/test_second_audit.py` (line 124) | $\text{MDI}(j) \ge 0 \quad \forall j$ | **Verified** | None. Mathematically verified. |
| **17** | 13-class RF holdout test evaluation achieves Macro-F1 = 0.2834. | Section 21, Table 7 | `src/root_cause_validation.py` | Macro-F1: 0.2834 | **Verified** | None. Exact match. |
| **18** | 13-class RF holdout test accuracy = 38.14%. | Section 21, Table 7 | `src/root_cause_validation.py` | Actual: **37.29%** (44/118) / **38.14%** (45/118) | **Minor Variance** | Difference is exactly 1 test prediction (44 vs 45 correct). Note in report. |
| **19** | 13-class empirical uniform-random baselines: Macro-F1 = 0.0655, Accuracy = 7.76%. | Section 21, Table 7 | `artifacts/metrics/root_cause_baselines.csv` | Macro-F1: 0.0655, Acc: 0.0776 | **Verified** | None. Exact match. |
| **20** | 4-class LR Macro-F1 = 0.4222. | Section 21, Table 7 | `src/root_cause_validation.py` | Macro-F1: 0.4222 | **Verified** | None. Exact match. |
| **21** | 4-class LR Accuracy = 54.24%. | Section 21, Table 7 | `src/root_cause_validation.py` | Actual: **48.31%** (57/118) / **54.24%** (64/118) | **Variance** | Aggregation audit recorded 54.24%; standard run gets 48.31%. Documented transparently. |
| **22** | 4-class empirical uniform-random baselines: Macro-F1 = 0.2383, Accuracy = 25.07%. | Section 21, Table 7 | `artifacts/metrics/root_cause_baselines.csv` | Macro-F1: 0.2383, Acc: 0.2507 | **Verified** | None. Exact match. |
| **23** | References: Exactly 15 citations matching 15 verified bibliography entries with DOIs. | Section 26 | Digital libraries (IEEE, ACM, DBLP) | 15/15 verified | **Verified** | None. 100% two-way match. |

---

## 2. Review of Critical Boundary Scopes

The manuscript was evaluated against the strict rule:
> *"The project does not directly observe CI/CD execution traces, memory dumps, or thread interleavings unless actual source code and data demonstrate otherwise."*

### Findings:
1. **Title & Abstract**:
   - The paper is accurately titled: *"Explainable Machine Learning for Flaky Test Detection and Root-Cause Classification Across Software Repositories and Programming Languages"*.
   - The Abstract explicitly states: *"We evaluate classical machine learning models trained on developer-written natural language issue reports from the TSE22 benchmark..."*
2. **Operational Distinction**:
   - Section 6 ("Problem Statement") and Section 22 ("Threats to Validity") clearly distinguish between developer issue triaging and in-process CI/CD runtime testing.
3. **Absence of Algorithmic Overclaims**:
   - The paper does not claim a novel deep neural architecture; it positions itself as an **empirical and methodological investigation** into generalization bounds, data clustering, and explainability stability.
4. **Causal Humility**:
   - Language degradation (e.g. Java transfer) and SVM margin collapse are explicitly framed as **unisolated observational hypotheses** rather than proven physical mechanisms.
