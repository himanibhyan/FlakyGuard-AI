# FlakyGuard AI: Complete Reproducibility Guide

This guide provides deterministic, step-by-step instructions to reproduce all experimental data, trained models, metrics CSVs, and publication figures from raw source data.

---

## 1. System and Hardware Requirements

| Specification | Minimum Requirement | Verified Test Platform |
| :--- | :--- | :--- |
| **Operating System** | macOS 12+, Ubuntu 20.04+, Windows 10/11 (WSL2) | macOS Darwin (ARM64 / Apple Silicon) |
| **Python** | 3.10, 3.11, 3.12, 3.13, 3.14 | Python 3.14.7 |
| **RAM** | 4 GB | 16 GB Unified Memory |
| **Disk Space** | 2 GB free disk space | 50 GB free disk space |
| **GPU / Accelerators** | None required (Classical ML CPU execution) | CPU Only |

---

## 2. Determinism and Random Seeds

All stochastic operations in FlakyGuard AI use explicitly pinned random seeds:
- **Global Seed**: `42`
- **Repeated Repository Partitions (`GroupShuffleSplit`)**: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`
- **Monte Carlo Baseline Trials**: `42` ($B=1,000$ iterations)
- **Bootstrap Confidence Intervals**: `42` ($B=1,000$ resamples)

---

## 3. End-to-End Pipeline Reproduction Script

Run each command in sequence from the project root directory with virtual environment activated:

```bash
# 0. Setup Environment Variables
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

# 1. Dataset Preprocessing & Canonicalization (~5 seconds)
python src/data_preprocessing.py

# 2. Exploratory Data Analysis & Diagnostic Figures (~8 seconds)
python src/eda_generator.py

# 3. Task A: In-Distribution Model Training & Evaluation (~10 seconds)
python src/model_training.py
python src/evaluation.py

# 4. Task B: Root-Cause Classification Training & Baselines (~12 seconds)
python src/root_cause_classifier.py
python src/root_cause_baselines.py
python src/root_cause_validation.py

# 5. Generalization Experiments: Repository & Cross-Language (~25 seconds)
python src/repeated_repository_evaluation.py
python src/lolo_evaluation.py

# 6. Statistical Bootstrap Analysis (~15 seconds)
python src/confidence_intervals.py

# 7. Explainability & Attribution Stability Audit (~10 seconds)
python src/explainability_audit.py

# 8. Regenerate All Publication Figures (~8 seconds)
python src/publication_figures.py

# 9. Execute Automated Test Suite (~3 seconds)
pytest tests/
```

**Total Execution Time**: $\approx 1.5$ to 2 minutes on a standard CPU.

---

## 4. Verification Checklist for Generated Artifacts

After running the pipeline above, verify that the following files exist and match reported values:

### Data Files (`data/processed/`):
- [x] `flaky_binary_dataset.csv` (1,101 rows; 701 Flaky, 400 False Positive; 641 canonical repos)
- [x] `root_cause_dataset.csv` (589 rows; 13 root-cause classes; 349 canonical repos)
- [x] `full_cleaned_dataset.csv` (1,513 rows; 842 canonical repos)

### Serialized Models (`models/`):
- [x] `flaky_vectorizer.joblib` (TF-IDF vectorizer, max 5,000 features)
- [x] `best_flaky_model.joblib` (Random Forest, 100 estimators)
- [x] `flaky_logistic_regression.joblib`, `flaky_linear_svm.joblib`, `flaky_random_forest.joblib`
- [x] `rc_vectorizer.joblib`, `best_rc_model.joblib`, `rc_label_encoder.joblib`

### Metric CSVs & Reports (`artifacts/metrics/`):
- [x] `binary_model_metrics.csv`: RF Macro-F1 = 0.6848, Accuracy = 72.40%
- [x] `repeated_repository_generalization.csv`: 30 rows across 10 seeds (LR Mean: 0.6670, RF Mean: 0.6542, SVM Mean: 0.5201)
- [x] `lolo_results.csv`: 15 rows across 5 languages and 3 models (LR Language Mean: 0.6053, RF Language Mean: 0.6002)
- [x] `root_cause_baselines.csv`: 1,000 Monte Carlo iterations for 13-class and 4-class baselines
- [x] `confidence_interval_report.md`: Cluster bootstrap SE inflation (+13% for RF, +4% for SVM, -2% for LR)
- [x] `explainability_claim_audit.md`: 9/10 sign consistency, `goroutine` sign flip

### Publication Figures (`artifacts/figures/publication/`):
- [x] `dataset_distribution.png`
- [x] `system_architecture.png`
- [x] `binary_model_comparison.png`
- [x] `repository_independent_comparison.png`
- [x] `repeated_repository_distribution.png`
- [x] `lolo_language_comparison.png`
- [x] `shap_global_explanation.png`
- [x] `explainability_stability.png`
- [x] `root_cause_confusion_matrix.png`
- [x] `root_cause_13_vs_4_comparison.png`

---

## 5. Automated Verification Test

Run the full verification test suite:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl pytest tests/ -v
```
All 27 test cases must pass without errors or warnings.
