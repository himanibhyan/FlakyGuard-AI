# FlakyGuard AI: Root-Cause Class Aggregation Audit

**Date**: November 2, 2026  
**Audited Transition**: 13 Empirical Categories $\rightarrow$ 4 Macroscopic Engineering Failure Modes  
**Population**: $N=589$ confirmed flaky test issue reports (Barbosa et al., IEEE TSE 2023)  
**Status**: Methodological Audit & Structural Validation Complete  

---

## 1. Executive Summary

In Task B (Root-Cause Classification), the original research taxonomy encompasses 13 fine-grained failure categories. Evaluation on these 13 classes yields a Macro-F1 of **0.2289** (Logistic Regression) to **0.2834** (Random Forest). In contrast, aggregating these 13 categories into 4 macroscopic engineering failure modes yields a Macro-F1 of **0.4222** (Accuracy: **54.24%**).

This audit investigates:
1. **Mapping Completeness & Integrity**: Whether the 13-to-4 mapping is exhaustive, bijective at the class-to-group level, and applied identically across training and testing partitions.
2. **Domain Grounding in Software Engineering Literature**: The theoretical rationale for the chosen groupings.
3. **Mathematical Mechanism Behind the Performance Rise**: Demonstrating that the increase from 0.2834 to 0.4222 is primarily driven by class space reduction and chance-baseline elevation, rather than superior model learning.
4. **Actionability vs Resolution Trade-Off**: Practical recommendations for production CI/CD triage.

---

## 2. Complete Mapping Verification & Sample Accounting

Every sample in the $N=589$ root-cause dataset was mapped according to the rules below. No records were discarded, omitted, or double-counted:

| Original Category (Barbosa et al.) | Dataset Count ($N=589$) | Dataset Share (%) | Test Support ($N=118$) | Assigned Macro-Category |
| :--- | :---: | :---: | :---: | :--- |
| **Async Wait** | 103 | 17.49% | 21 | **Timing & Asynchrony** |
| **Time** | 28 | 4.75% | 6 | **Timing & Asynchrony** |
| **Test Case Timeout** | 28 | 4.75% | 5 | **Timing & Asynchrony** |
| **Test Suite Timeout** | 5 | 0.85% | 1 | **Timing & Asynchrony** |
| *Subtotal: Timing & Asynchrony* | *164* | *27.84%* | *33* | |
| **Concurrency** | 97 | 16.47% | 19 | **Concurrency & Non-Determinism** |
| **Unordered Collections** | 28 | 4.75% | 6 | **Concurrency & Non-Determinism** |
| **Randomness** | 22 | 3.74% | 4 | **Concurrency & Non-Determinism** |
| *Subtotal: Concurrency & Non-Determinism* | *147* | *24.96%* | *29* | |
| **Platform Dependency** | 86 | 14.60% | 17 | **Resource & Environment** |
| **Network** | 71 | 12.05% | 14 | **Resource & Environment** |
| **Resource Leak** | 61 | 10.36% | 12 | **Resource & Environment** |
| *Subtotal: Resource & Environment* | *218* | *37.01%* | *43* | |
| **Test Order Dependency** | 40 | 6.79% | 8 | **Test Logic & Assertions** |
| **Floating Point Operations** | 14 | 2.38% | 3 | **Test Logic & Assertions** |
| **Too Restrictive Range** | 6 | 1.02% | 2 | **Test Logic & Assertions** |
| *Subtotal: Test Logic & Assertions* | *60* | *10.19%* | *13* | |
| **Total** | **589** | **100.00%** | **118** | **4 Distinct Classes** |

### Split Verification:
- **Partitioning**: Stratified 80/20 train/test split (`random_state=42`).
- **Train partition**: $N=471$ samples (Timing: 131, Concurrency: 118, Environment: 175, Test Logic: 47).
- **Test partition**: $N=118$ samples (Timing: 33, Concurrency: 29, Environment: 43, Test Logic: 13).
- **Integrity**: Mapping was executed identically on the full dataset before splitting, ensuring 100% feature and label alignment without leakage.

---

## 3. Justification via Software Testing Literature

The macroscopic categorization is directly grounded in landmark software engineering empirical studies on flaky tests:

1. **Timing & Asynchrony**:
   - *Literature Basis*: Luo et al. (ACM FSE 2014, "An Empirical Analysis of Flaky Tests") and Bell et al. (ICSE 2018, "DeFlaker") establish that `Async Wait`, explicit sleep statements (`Time`), and timeouts (`Test Case Timeout`, `Test Suite Timeout`) stem from identical underlying test smell: assuming fixed wall-clock execution speed for asynchronous background events.
   - *Vocabulary Overlap*: Issue descriptions consistently conflate these terms (e.g., "test timed out after 5000ms waiting for callback").
2. **Concurrency & Non-Determinism**:
   - *Literature Basis*: Lam et al. (ICST 2019) and Gruber et al. (ASE 2021) observe that thread interleavings (`Concurrency`), hash code iteration order (`Unordered Collections`), and pseudorandom generators (`Randomness`) all represent non-deterministic state space exploration during parallel or multi-threaded execution.
3. **Resource & Environment**:
   - *Literature Basis*: Gao et al. (IEEE TSE 2021) group socket/HTTP failures (`Network`), unclosed file handles/memory exhaustion (`Resource Leak`), and OS/architecture discrepancies (`Platform Dependency`) into *environmental and external infrastructure fragility*, where flakiness originates outside the test logic itself.
4. **Test Logic & Assertions**:
   - *Literature Basis*: Gyori et al. (FSE 2015, "NonDex") and Zhang et al. (ISSTA 2014) demonstrate that shared mutable static state between tests (`Test Order Dependency`), numerical rounding variations across architectures (`Floating Point Operations`), and overly narrow assertion thresholds (`Too Restrictive Range`) constitute algorithmic errors in the test specification.

---

## 4. Deconstructing the Metric Rise: Mechanical Artifact vs Algorithmic Gain

The exploratory 4-class classifier achieves **0.4222** Macro-F1 compared to **0.2289** for 13-class Logistic Regression and **0.2834** for 13-class Random Forest. It is methodologically impermissible to claim this represents an improved algorithm. The rise is explained by three mathematical and structural mechanisms:

### A. The Baseline Shift (Denominator Effect)
In multi-class evaluation, random chance and majority-class baselines scale inversely with the number of classes $K$:

| Metric | 13-Class Setting ($K=13$) | 4-Class Setting ($K=4$) | Baseline Shift |
| :--- | :---: | :---: | :---: |
| **Uniform Random Accuracy Baseline** | $1/13 = \mathbf{7.69\%}$ | $1/4 = \mathbf{25.00\%}$ | **+17.31 pp** ($3.25\times$) |
| **Majority Prior Baseline (Zero-Rule)** | $103/589 = \mathbf{17.49\%}$ | $218/589 = \mathbf{37.01\%}$ | **+19.52 pp** ($2.12\times$) |
| **Expected Uniform Macro-F1** | $\approx \mathbf{0.0769}$ | $\approx \mathbf{0.2500}$ | **+0.1731** |

When evaluated relative to the random baseline:
- 13-Class Random Forest (Macro-F1: 0.2834): **$3.69\times$ over random baseline** ($0.2834 / 0.0769$).
- 4-Class Logistic Regression (Macro-F1: 0.4222): **$1.69\times$ over random baseline** ($0.4222 / 0.2500$).

> [!NOTE]
> Relative to the chance baseline, the 13-class classifier actually captures a higher multiple of information beyond random guessing than the 4-class classifier.

### B. Penalty Elimination for Within-Cluster Confusion
In the 13-class problem, an issue with `root_cause = "Async Wait"` that mentions "timeout after 5 seconds" will often be predicted as `"Test Case Timeout"`. 
- In the 13-class evaluation, this prediction incurs **two penalties**: a False Negative for `Async Wait` and a False Positive for `Test Case Timeout`.
- In the 4-class evaluation, both categories map to `Timing & Asynchrony`. The prediction is counted as a **True Positive**, mechanically increasing precision, recall, and F1.

### C. Resolution of Extreme Minority Class Penalty
In the 13-class test set ($N=118$):
- `Test Suite Timeout` has only $N=1$ test sample.
- `Too Restrictive Range` has only $N=2$ test samples.
- `Floating Point Operations` has only $N=3$ test samples.

If the model predicts 0 true positives for `Test Suite Timeout` (as is common when trained on only 4 samples), the F1 for that class is identically 0. In an unweighted Macro-F1 calculation:
$$\text{Macro-F1} = \frac{1}{13} \sum_{k=1}^{13} F_1^{(k)}$$
A single class with $F_1 = 0$ reduces the overall score by $\frac{1}{13} \approx 0.0769$. When 4 rare classes score near zero, the theoretical maximum Macro-F1 across the remaining 9 classes is compressed to $\leq 0.692$. In the 4-class grouping, all 4 macro classes have substantial support in the test set ($N=33, 29, 43, 13$), eliminating zero-support degradation.

---

## 5. Comparative Evaluation Summary

| Architecture | Class Space ($K$) | Macro-F1 | Weighted-F1 | Accuracy | Random Baseline | Ratio over Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Balanced)** | 13 | 0.1844 | 0.3340 | 33.90% | 7.69% | $2.40\times$ |
| **Logistic Regression (Balanced)** | 13 | 0.2289 | 0.3541 | 36.44% | 7.69% | $2.98\times$ |
| **Random Forest (Balanced)** | 13 | 0.2834 | 0.3705 | 38.14% | 7.69% | $\mathbf{3.69\times}$ |
| **Logistic Regression (Balanced)** | 4 | **0.4222** | **0.5376** | **54.24%** | 25.00% | $1.69\times$ |

---

## 6. Scientific Recommendation & Deployment Guidance

1. **Do Not Claim Algorithmic Superiority**: Reports and papers must explicitly distinguish between the 13-class benchmark (confirmatory research evaluation following Barbosa et al.) and the 4-class aggregation (exploratory operational triage).
2. **Dual-Tier Deployment in CI/CD**:
   - **Tier 1 (Triage Router)**: Deploy the 4-class model to route incoming flaky issues to responsible teams (e.g., Infrastructure Team for *Resource & Environment*; Core Platform for *Timing & Asynchrony*).
   - **Tier 2 (Specialist Diagnostic)**: Deploy the 13-class model with top-3 candidate suggestions and explainability highlights for developers performing detailed code fixes.
