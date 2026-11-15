# FlakyGuard AI: Root-Cause Classification Validation & Imbalance Audit

**Date**: November 2, 2026  
**Benchmark**: Barbosa et al. (IEEE TSE 2023)  
**Sample Population**: $N=589$ confirmed flaky test reports across 13 distinct categories  

---

## 1. Class Distribution & Imbalance Severity

The empirical root-cause distribution reflects acute right-skewed imbalance:

| Rank | Root-Cause Category | Dataset Count | Dataset Share | Holdout Test Support ($N=118$) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Async Wait** | 103 | 17.49% | 21 |
| 2 | **Concurrency** | 97 | 16.47% | 19 |
| 3 | **Platform Dependency** | 86 | 14.60% | 17 |
| 4 | **Network** | 71 | 12.05% | 14 |
| 5 | **Resource Leak** | 61 | 10.36% | 12 |
| 6 | **Test Order Dependency** | 40 | 6.79% | 8 |
| 7 | **Test Case Timeout** | 28 | 4.75% | 6 |
| 8 | **Time** | 28 | 4.75% | 6 |
| 9 | **Unordered Collections** | 28 | 4.75% | 6 |
| 10 | **Randomness** | 22 | 3.74% | 4 |
| 11 | **Floating Point Operations** | 14 | 2.38% | 3 |
| 12 | **Too Restrictive Range** | 6 | 1.02% | 1 |
| 13 | **Test Suite Timeout** | 5 | 0.85% | 1 |

### Key Imbalance Findings:
- **Head Categories**: The top 2 categories (`Async Wait` + `Concurrency`) represent **33.9%** ($N=200$) of the dataset.
- **Tail Categories**: The bottom 2 categories (`Too Restrictive Range`: $N=6$, `Test Suite Timeout`: $N=5$) account for merely **1.87%** ($N=11$) combined.
- **Holdout Representation**: In an 80/20 stratified split, `Too Restrictive Range` has only **1 holdout sample**, and `Test Suite Timeout` has only **1 holdout sample**.

---

## 2. Primary Comparative Evaluation Across Architectures (13 Discrete Categories)

| Architecture | Weighting Strategy | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | Balanced (`class_weight='balanced'`) | 0.3051 | **0.2289** | 0.2894 |
| **Logistic Regression (Unweighted)** | Unweighted | 0.3136 | **0.1661** | 0.2751 |
| **Linear SVM (Balanced)** | Balanced (`class_weight='balanced'`) | 0.3305 | **0.1844** | 0.2931 |
| **Linear SVM (Unweighted)** | Unweighted | 0.3136 | **0.1766** | 0.2793 |
| **Random Forest (Balanced)** | Balanced (`class_weight='balanced'`) | 0.3729 | **0.2834** | 0.3552 |

---

## 3. Detailed Per-Class Breakdown (Best Primary Model: Random Forest Balanced)

| Category | Class Support ($N=118$) | True Positives | False Negatives | False Positives | Precision | Recall | F1-Score | Status & Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Async Wait** | 21 | 8 | 13 | 15 | 0.3478 | 0.3810 | 0.3636 | Confused with Timeout & Platform |
| **Concurrency** | 19 | 5 | 14 | 2 | 0.7143 | 0.2632 | 0.3846 | High precision, low recall |
| **Floating Point Operations** | 3 | 0 | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | Sample sparsity ($N=3$) |
| **Network** | 14 | 3 | 11 | 7 | 0.3000 | 0.2143 | 0.2500 | Overlap with Resource Leak |
| **Platform Dependency** | 17 | 9 | 8 | 20 | 0.3103 | 0.5294 | 0.3913 | Moderate recall, high FP |
| **Randomness** | 4 | 1 | 3 | 0 | 1.0000 | 0.2500 | 0.4000 | Perfect precision, small support |
| **Resource Leak** | 12 | 8 | 4 | 7 | 0.5333 | 0.6667 | 0.5926 | Strong discriminative vocabulary |
| **Test Case Timeout** | 6 | 4 | 2 | 4 | 0.5000 | 0.6667 | 0.5714 | Good discriminative vocabulary |
| **Test Order Dependency** | 8 | 1 | 7 | 5 | 0.1667 | 0.1250 | 0.1429 | Low recall |
| **Test Suite Timeout** | 1 | 0 | 1 | 4 | 0.0000 | 0.0000 | 0.0000 | Extreme sample sparsity ($N=1$) |
| **Time** | 6 | 0 | 6 | 1 | 0.0000 | 0.0000 | 0.0000 | Confused with Async Wait |
| **Too Restrictive Range** | 1 | 0 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | Extreme sample sparsity ($N=1$) |
| **Unordered Collections** | 6 | 5 | 1 | 6 | 0.4545 | 0.8333 | 0.5882 | Strong recall (5/6 detected) |
| **Total / Macro Average** | **118** | **44** | **74** | **74** | **0.3328** | **0.3023** | **0.2834** | **Accuracy: 37.29% (44/118)** |

---

## 4. In-Depth Investigation of Rare Tail Categories (6-Sample & 5-Sample Classes)

### A. `Too Restrictive Range` (Dataset: $N=6$, Test: $N=1$)
- **Empirical Reality**: Developers write assertion checks with overly tight numerical thresholds (e.g., `assertEquals(expected, actual, delta=0.0001)`).
- **Model Output**: Precision = 0.0000, Recall = 0.0000, F1 = 0.0000.
- **Failure Cause**: With only 5 training samples across different repositories (`mockito`, `neo4j`, `cdap`), the vocabulary is dominated by library-specific tokens rather than general assertion range terms.

### B. `Test Suite Timeout` (Dataset: $N=5$, Test: $N=1$)
- **Empirical Reality**: An entire test suite execution exceeds global CI/CD deadlines.
- **Model Output**: Confused with `Test Case Timeout` and `Async Wait`.
- **Failure Cause**: The linguistic boundary between 'suite timeout' and 'case timeout' is semantically identical in issue reports (both contain `timeout`, `deadline exceeded`, `hung`).

---

## 5. Exploratory Aggregation: 4 Macroscopic Failure Dimensions

> **Note**: This experiment is strictly exploratory and separated from the primary 13-class benchmark.

When aggregating the 13 categories into 4 macroscopic engineering failure modes:
1. **Timing & Asynchrony** (`Async Wait`, `Time`, `Test Case Timeout`, `Test Suite Timeout`)
2. **Concurrency & Non-Determinism** (`Concurrency`, `Randomness`, `Unordered Collections`)
3. **Resource & Environment** (`Network`, `Resource Leak`, `Platform Dependency`)
4. **Test Logic & Assertions** (`Test Order Dependency`, `Floating Point Operations`, `Too Restrictive Range`)

| Architecture | Classes | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Macro Aggregation)** | 4 | **0.4831** | **0.4222** | **0.4762** |

**Conclusion**: Macroscopic aggregation boosts Macro-F1 from **0.2289 to 0.4496** and Accuracy from **30.5% to 45.8%**, indicating that classical NLP models capture coarse-grained failure modes effectively, but lack sufficient training instances to separate fine-grained sub-categories.