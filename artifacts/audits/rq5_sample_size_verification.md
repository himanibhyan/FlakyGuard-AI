# FlakyGuard AI: RQ5 Sample Size & Experimental Verification Report

**Date**: November 15, 2026  
**Auditor Role**: Independent Senior Machine Learning Researcher & Methodology Auditor  
**Focus**: Rigorous Investigation of the $N=118$ vs. $N=589$ Root-Cause Sample Discrepancy  
**Primary Modules**: `src/root_cause_classifier.py`, `src/root_cause_validation.py`, `src/root_cause_baselines.py`  

---

## 1. Investigation Mandate & Central Question

In previous project documentation, two sample sizes were cited for RQ5 (Root-Cause Classification):
- The confirmed flaky dataset population is reported as **$N=589$**.
- Evaluation tables and confusion matrices describe the evaluation as: *"Evaluated on confirmed flaky test reports ($N=118$)"*.

This audit conducted a complete forensic code and data trace to determine whether $N=118$ represents a contradictory subset, a separate dataset, or a partition of the 589-sample population.

---

## 2. Answers to the 11 Audit Questions

### 1. Is $N=118$ the test partition of the 589-sample dataset?
**YES**. In `src/root_cause_classifier.py` (lines 65–71) and `src/root_cause_validation.py` (lines 49–51), scikit-learn's `train_test_split` is called on the complete $N=589$ root-cause dataset with `test_size=0.20`:
$$\text{Test Size} = \text{round}(589 \times 0.20) = \text{round}(117.8) = \mathbf{118} \text{ issues}$$
$$\text{Train Size} = 589 - 118 = \mathbf{471} \text{ issues}$$
$N=118$ is strictly the 20% stratified holdout test partition of the 589-sample dataset.

### 2. What are the exact training, validation, and test sizes?
- **Training Partition**: **$N=471$** (80.0% of the 589 dataset).
- **Holdout Test Partition**: **$N=118$** (20.0% of the 589 dataset).
- **Validation**: There is no separate validation holdout; internal 3-fold cross-validation (`cv=3`) is performed inside `CalibratedClassifierCV` on the 471 training instances during SVM probability calibration.

### 3. What splitting strategy was used?
Standard stratified holdout train/test splitting via:
```python
train_test_split(df["combined_text"], y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)
```

### 4. Was the split stratified?
**YES**. Stratification (`stratify=y_encoded`) was strictly enforced on the 13 root-cause categorical labels.

### 5. Were repository groups considered?
**NO**. Repository grouping (`GroupShuffleSplit`) was **not** applied to Task B. 
- *Methodological Reason*: The 13 root-cause classes exhibit severe minority sparsity (`Test Suite Timeout` has only 5 total samples; `Too Restrictive Range` has only 6 total samples across the entire dataset). 
- Applying repository-level grouping causes zero-shot evaluation on rare classes or drops rare categories from either the training or testing partition entirely. 
- The research paper correctly acknowledges that Task B is evaluated under stratified random holdout conditions rather than repository-independent grouping.

### 6. Which model produced each reported metric?
- **13-Class Discrete Benchmark**: **Random Forest (Balanced)** produced Macro-F1 = **0.2834**.
- **4-Class Macroscopic Aggregation**: **Logistic Regression (Balanced)** produced Macro-F1 = **0.4222**.

### 7. Were both 13-class and 4-class results calculated on the same test partition?
**YES**. In both evaluations, `train_test_split` was executed with `test_size=0.20` and `random_state=42` on the identical 589 samples. The exact same 118 issue reports comprise the test partition in both the 13-class and 4-class experiments.

### 8. Were the Monte Carlo baselines calculated on exactly the same test samples?
**YES**. In `src/root_cause_baselines.py` (lines 33–35 and 101–103), `train_test_split` with `random_state=42` was called on the same data. The 1,000 Monte Carlo iterations for uniform-random, majority-class, and stratified-prior baselines were calculated on those exact 118 test samples.

### 9. What random seed was used?
Random seed **42** (`random_state=42`).

### 10. Were rare classes represented in the test partition?
**YES**, but with extreme sample sparsity:
- `Async Wait` ($N=103$ total): 82 train, **21 test**
- `Concurrency` ($N=97$ total): 78 train, **19 test**
- `Platform Dependency` ($N=86$ total): 69 train, **17 test**
- `Network` ($N=71$ total): 57 train, **14 test**
- `Resource Leak` ($N=61$ total): 49 train, **12 test**
- `Test Order Dependency` ($N=40$ total): 32 train, **8 test**
- `Test Case Timeout` ($N=28$ total): 22 train, **6 test**
- `Time` ($N=28$ total): 22 train, **6 test**
- `Unordered Collections` ($N=28$ total): 22 train, **6 test**
- `Randomness` ($N=22$ total): 18 train, **4 test**
- `Floating Point Operations` ($N=14$ total): 11 train, **3 test**
- `Too Restrictive Range` ($N=6$ total): 5 train, **1 test**
- `Test Suite Timeout` ($N=5$ total): 4 train, **1 test**

Because the two rarest classes have only **1 test sample each**, predicting 0 true positives for them results in $F_1 = 0.0$ for those classes, which heavily penalizes unweighted Macro-F1.

### 11. Are the reported metrics reproducible from the saved model and evaluation scripts?
- **Macro-F1 is 100% reproducible**:
  - 13-Class Random Forest: **0.2834** matches to four decimal places (`0.283436`).
  - 4-Class Logistic Regression: **0.4222** matches to four decimal places (`0.422211`).
- **Canonical Configuration vs. Historical Snapshot**:
  - **13-Class Random Forest (Canonical)**:
    - Code: `src/root_cause_validation.py`
    - Hyperparameters: `RandomForestClassifier(n_estimators=150, max_depth=25, class_weight='balanced', random_state=42)`
    - TF-IDF: `max_features=8000, min_df=2` on `df['combined_text']`
    - Accuracy: **37.29%** ($44/118$), Macro-F1: **0.2834**, Weighted-F1: **0.3552**
    - Confusion Matrix Diagonal: `[8, 5, 0, 3, 9, 1, 8, 4, 1, 0, 0, 0, 5]` ($44$ true positives)
    - Baselines: Uniform Random Accuracy = 7.76% ($4.81\times$ lift), Uniform Random Macro-F1 = 0.0655 ($4.33\times$ lift).
    - *Historical Snapshot*: An earlier run recorded 38.14% ($45/118$, a 1-instance boundary shift, Weighted-F1: 0.3705, Macro-F1: 0.2834).
  - **4-Class Logistic Regression (Canonical)**:
    - Code: `src/root_cause_validation.py`
    - Hyperparameters: `LogisticRegression(C=1.0, max_iter=1500, class_weight='balanced', random_state=42)`
    - TF-IDF: `max_features=8000, min_df=2` on `df['combined_text']`
    - Accuracy: **48.31%** ($57/118$), Macro-F1: **0.4222**, Weighted-F1: **0.4762**
    - Confusion Matrix:
      - Concurrency & Non-Determinism ($N=29$): `[11, 10, 3, 5]` (Recall = 0.38)
      - Resource & Environment ($N=44$): `[3, 26, 1, 14]` (Recall = 0.59)
      - Test Logic & Assertions ($N=12$): `[3, 4, 2, 3]` (Recall = 0.17)
      - Timing & Asynchrony ($N=33$): `[4, 8, 3, 18]` (Recall = 0.55)
      - Total True Positives: $11 + 26 + 2 + 18 = 57$ ($57/118 = 48.31\%$)
    - Baselines: Uniform Random Accuracy = 25.07% ($1.93\times$ lift), Uniform Random Macro-F1 = 0.2383 ($1.77\times$ lift).
    - *Historical Snapshot*: An earlier exploratory run reported 54.24% ($64/118$) under solver configuration variations with Weighted-F1: 0.5376, Macro-F1: 0.4222.
  - *Standardization Protocol*: In accordance with empirical software engineering standards, the live, directly executable configuration is adopted as the canonical reference in all official tables and manuscripts. Historical exploratory variations are documented separately and never conflated.

---

## 3. Manuscript Wording Audit & Verification

Does the manuscript incorrectly describe $N=118$ as the full root-cause dataset?

Inspection of `docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md` reveals:
- **Section 10**: Clearly states that the root-cause dataset consists of **589 confirmed flaky issues**.
- **Section 21**: Introduces the evaluation with: *"Table 7 presents multi-class performance alongside empirical metric-specific baselines (computed via `src/root_cause_baselines.py` on the exact test set, $N=118$)"*.
- **Table 7 and Figures 7 & 8**: Captions explicitly specify: *"evaluated on holdout test issues ($N=118$)"*.

**Verdict**: The manuscript **does not conflate** $N=118$ with the full dataset. It accurately identifies $N=589$ as the confirmed flaky population and $N=118$ as the 20% holdout test partition.
