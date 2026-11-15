# FlakyGuard AI: Complete Execution and Operational Guide

This document provides exact, end-to-end instructions for setting up the environment, running data pipelines, training models, reproducing research experiments, running inference via CLI, and launching the Streamlit web dashboard.

---

## 1. Environment Setup

### Prerequisites
- **Operating System**: macOS, Linux, or Windows (WSL recommended)
- **Python Version**: Python 3.10 – 3.14
- **Hardware**: Standard multi-core CPU (8GB+ RAM recommended; no GPU required)

### Step 1: Clone Repository
```bash
git clone <repository_url>
cd "AI-Based Flaky Test Detection"
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows: `.\.venv\Scripts\activate`)*

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Always execute python commands with `PYTHONPATH=.` and `MPLCONFIGDIR=/tmp/mpl`:
```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

---

## 2. Dataset Processing Pipeline

To regenerate all cleaned and canonicalized datasets from raw data:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/data_preprocessing.py
```

### What this does:
1. Ingests `data/raw/DataBase_GitHub.xlsx` (Sheet: `Data`).
2. Deduplicates records by URL and text body (1,543 $\rightarrow$ 1,513).
3. Resolves canonical `{owner}/{repo}` identifiers across 641 distinct repositories.
4. Generates:
   - `data/processed/flaky_binary_dataset.csv` ($N=1,101$)
   - `data/processed/root_cause_dataset.csv` ($N=589$)
   - `data/processed/full_cleaned_dataset.csv` ($N=1,513$)
5. Outputs dataset summary JSON to `artifacts/metrics/preprocessing_summary.json`.

---

## 3. Model Training & Persisting

To train and serialize the binary flakiness models and root-cause classification models:

```bash
# 1. Train Task A (Binary Flakiness Detection) Models
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/model_training.py

# 2. Train Task B (Root-Cause Taxonomy Classification) Models
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/root_cause_classifier.py
```

### Generated Artifacts:
- `models/flaky_vectorizer.joblib`: Fitted TF-IDF vectorizer (Task A)
- `models/best_flaky_model.joblib`: Serialized Random Forest model
- `models/flaky_random_forest.joblib`, `models/flaky_logistic_regression.joblib`, `models/flaky_linear_svm.joblib`
- `models/rc_vectorizer.joblib`: Fitted TF-IDF vectorizer (Task B)
- `models/best_rc_model.joblib`: Serialized Root Cause model
- `models/rc_label_encoder.joblib`: 13-class categorical label encoder

---

## 4. Reproducing Research Experiments

### RQ1: In-Distribution Evaluation
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/evaluation.py
```
*Evaluates Logistic Regression, Random Forest, and Linear SVM on the stratified holdout test set ($N=221$). Outputs metrics to `artifacts/metrics/binary_model_metrics.csv`.*

### RQ2: Repository-Independent Generalization (10 Repeated Partitions)
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/repeated_repository_evaluation.py
```
*Executes `GroupShuffleSplit` across 10 random seeds (`[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`). Enforces strictly zero repository overlap. Outputs `artifacts/metrics/repeated_repository_generalization.csv`.*

### RQ3: Cross-Language Generalization (Leave-One-Language-Out)
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/lolo_evaluation.py
```
*Trains on 4 languages and tests on the held-out 5th language across C, Go, Java, JavaScript, Python. Outputs `artifacts/metrics/lolo_results.csv`.*

### RQ4: Explainability & Attribution Stability Audit
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/explainability_audit.py
```
*Audits token directional stability between random-split and repository-independent models; computes SHAP values and LIME local surrogates. Outputs `artifacts/metrics/explainability_claim_audit.md`.*

### RQ5: Root-Cause Granularity & Empirical Baselines
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/root_cause_baselines.py
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/root_cause_validation.py
```
*Runs 1,000 Monte Carlo baseline trials on the test set ($N=118$) for both 13-class and 4-class settings. Outputs `artifacts/metrics/root_cause_baselines.csv` and `artifacts/metrics/root_cause_generalization_report.md`.*

### Statistical Cluster Bootstrap Analysis
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/confidence_intervals.py
```
*Runs $B=1,000$ non-parametric IID and cluster bootstrap trials. Outputs `artifacts/metrics/confidence_interval_report.md`.*

### Regenerate All Publication Figures
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/publication_figures.py
```
*Generates 10 publication-quality 300 DPI figures in `artifacts/figures/publication/`.*

---

## 5. Command-Line Inference (CLI)

FlakyGuard AI provides a standalone command-line inference tool via `src/inference.py`:

### Basic Usage:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/inference.py \
  --title "TestNetworkTimeout is intermittent on CI" \
  --description "Socket connection timed out after 5000ms waiting for server heartbeat" \
  --language "Python"
```

### Full Diagnostic with Root-Cause Prediction:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/inference.py \
  --title "Race condition in goroutine worker pool" \
  --description "Intermittent deadlock when closing channels concurrently under load" \
  --language "Go" \
  --root-cause
```

### JSON Output for Pipeline Integration:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/inference.py \
  --title "Async await timeout" \
  --description "Promise rejected due to timer expiration" \
  --language "JavaScript" \
  --root-cause \
  --json
```

### Model Override Option:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python src/inference.py \
  --title "Thread safety issue" \
  --description "Shared resource modified without synchronization" \
  --language "Java" \
  --model "Logistic Regression"
```

---

## 6. Interactive Streamlit Dashboard

To launch the multi-page Streamlit web user interface:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Dashboard Navigation:
1. **Executive Overview (`app.py`)**: System architecture, key research findings, dataset provenance.
2. **Overview Page (`pages/1_Overview.py`)**: Comprehensive research problem statement, research questions, and experimental methodology.
3. **Flaky Test Detector (`pages/2_Flaky_Test_Detector.py`)**: Live diagnostic input form with probability gauges, influential token highlights, and root-cause predictions.
4. **Model Evaluation (`pages/3_Model_Evaluation.py`)**: Interactive performance metric tables, ROC/PR curves, confusion matrices, and empirical baselines.
5. **Explainability (`pages/4_Explainability.py`)**: Global SHAP feature importance charts, LIME diagnostic case studies, and token stability under distribution shift.
6. **Dataset Explorer (`pages/5_Dataset_Explorer.py`)**: Filterable table of 1,101 issues by language, flakiness status, repository, and search keywords.

---

## 7. Running the Automated Test Suite

Run the full pytest suite:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl pytest tests/
```

### Expected Output:
```text
============================== 27 passed in 2.30s ==============================
```

To run with verbose test output:
```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl pytest -v tests/
```
