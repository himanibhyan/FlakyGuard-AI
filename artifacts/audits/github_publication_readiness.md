# FlakyGuard AI: GitHub Publication Readiness & Repository Security Audit

**Date**: November 15, 2026  
**Auditor Role**: Independent Senior Machine Learning Researcher, Software Auditor & Security Specialist  
**Current Git State**: Initial Root Commit `4bb1051` on branch `main` (149 committed files)  
**Configured Remotes**: None (Push intentionally withheld pending audit review)  

---

## 1. Executive Summary & Verdict

### **VERDICT: READY AFTER SPECIFIC FIXES (HIGH CONFIDENCE)**

The FlakyGuard AI codebase is technically mature, fully functional, and methodologically documented. All 27 automated tests pass, models execute in milliseconds, and the working tree is clean. 

However, before executing a public `git push` to an open GitHub repository, **two specific data privacy and licensing actions must be resolved**:
1. **Developer Email Sanitization in Raw Scraped Files**: The raw scraped cache files (`data/raw/raw_flaky_issues.csv` and `data/raw/github_issues_metadata.json`) contain 80 unanonymized developer email addresses extracted from issue bodies. These raw cache files should either be sanitized or excluded from public Git tracking. The cleaned datasets in `data/processed/` are already completely sanitized (0 emails found).
2. **Dataset Provenance & Redistribution Notice**: While `DataBase_GitHub.xlsx` originates from the publicly accessible IEEE TSE 2023 replication package (`Test-Flaky/TSE22`), a clear academic attribution notice and reproduction command (`src/data_acquisition.py`) must accompany the repository.

---

## 2. Dataset Licensing and Privacy Audit

### 2.1 Provenance of `DataBase_GitHub.xlsx`
- **Origin**: Replication repository for *Test Flakiness Across Programming Languages* (Barbosa et al., *IEEE TSE 2023*), hosted at `https://github.com/Test-Flaky/TSE22`.
- **Content**: Contains issue IDs, project names, language tags, URL links, and reviewer categorization annotations.
- **Privacy Check**: Scanned across all 9 sheets for email patterns: **0 email addresses found**. It is safe for academic distribution under standard research citation terms.

### 2.2 Scraped Issue Caches (`raw_flaky_issues.csv` & `github_issues_metadata.json`)
- **Origin**: Scraped by `src/data_acquisition.py` from public GitHub issue and pull request pages to retrieve title and body text.
- **Privacy Finding**: Scanned with regex `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`:
  - `data/raw/raw_flaky_issues.csv`: **80 developer email addresses detected** (e.g. `[REDACTED_EMAIL]`, `[REDACTED_EMAIL]`).
  - `data/raw/github_issues_metadata.json`: **80 developer email addresses detected**.
- **Processed Datasets Check (`data/processed/`)**:
  - `flaky_binary_dataset.csv`: **0 email addresses detected**.
  - `root_cause_dataset.csv`: **0 email addresses detected**.
  - Text normalization in `src/data_preprocessing.py` (line 32: `re.sub(r"\S+@\S+", " ", text)`) successfully stripped all emails from the processed modeling datasets.
- **Action Required**: Add `data/raw/github_issues_metadata.json` and `data/raw/raw_flaky_issues.csv` to `.gitignore` or sanitize emails before making the GitHub repository public.

---

## 3. Serialized Model Artifacts Audit (`models/`)

The repository contains 12 serialized `.joblib` model artifacts totaling **7.74 MB**:

| File Name | Size | Type / Algorithm | Contains Embedded Raw Text? | Essential for Demo? |
| :--- | :---: | :--- | :---: | :---: |
| `best_flaky_model.joblib` | 2.38 MB | Random Forest (150 trees) | **No** (Trees store numerical thresholds) | **Yes** (Task A) |
| `flaky_random_forest.joblib` | 2.38 MB | Random Forest (150 trees) | **No** | Yes |
| `rc_linear_svm.joblib` | 1.53 MB | Calibrated Linear SVM | **No** (Stores hyperplane weights) | Yes |
| `best_rc_model.joblib` | 511 KB | Balanced Logistic Regression | **No** (Stores weight matrix) | **Yes** (Task B) |
| `rc_logistic_regression.joblib`| 511 KB | Balanced Logistic Regression | **No** | Yes |
| `flaky_vectorizer.joblib` | 338 KB | Scikit-learn TF-IDF Vectorizer | **No** (Vocabulary dict + IDF array) | **Yes** (Task A) |
| `flaky_linear_svm.joblib` | 211 KB | Calibrated Linear SVM | **No** | Yes |
| `rc_vectorizer.joblib` | 188 KB | Scikit-learn TF-IDF Vectorizer | **No** | **Yes** (Task B) |
| `flaky_logistic_regression.joblib`| 71 KB | Logistic Regression | **No** | Yes |
| `rc_classification_meta.joblib`| 4 KB | Metadata dictionary | **No** | Yes |
| `flaky_detection_meta.joblib`| 1.3 KB | Metadata dictionary | **No** | Yes |
| `rc_label_encoder.joblib` | 709 B | Scikit-learn LabelEncoder | **No** (String class names only) | **Yes** (Task B) |

### Key Findings:
1. **No Embedded Training Text**: None of the models serialize raw developer issue descriptions or personally identifiable strings.
2. **File Size Compliance**: All individual artifacts are $<2.5\text{ MB}$, far below GitHub's 100 MB hard limit and 50 MB recommendation threshold.
3. **Reproducibility**: Every model can be regenerated from source in $<30\text{ seconds}$ by executing `python src/model_training.py` and `python src/root_cause_classifier.py`.

---

## 4. Repository Security, Credentials & Secrets Scan

- **Automated Secrets Scan**: Executed regex scan across the entire repository for API keys (`api_key`, `apikey`), passwords, Bearer tokens, GitHub personal access tokens (`ghp_`), and cloud credentials:
  - **Result: 0 secrets, 0 credentials, 0 tokens detected.**
- **Local Absolute Path Scan**: Executed search for `/Users/himani` across all Python source files (`src/`, `pages/`, `app.py`, `tests/`):
  - **Result: 0 hardcoded absolute paths in executable source code.**
  - All markdown figure references use relative paths (`artifacts/figures/publication/...` or `../../artifacts/figures/...`) ensuring clean rendering in GitHub web previews.
- **Temporary Cache Cleanup**: Verified that `.gitignore` excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`, and `.streamlit/credentials.toml`.

---

## 5. README & Setup Instruction Verification

- The README instructions were audited against a clean virtual environment:
  - Setup commands (`python3 -m venv .venv`, `pip install -r requirements.txt`) work cleanly.
  - CLI inference executes instantly (`python src/inference.py --title "..."`).
  - Streamlit dashboard launches on `http://localhost:8501`.
  - All 27 tests pass in 2.67 seconds (`pytest tests/`).
- The execution time of the reproduction pipeline was timed:
  - Data preprocessing: 5.2 seconds
  - Binary model training: 4.8 seconds
  - Root cause training: 3.1 seconds
  - LOLO evaluation (5 languages): 18.4 seconds
  - Repeated partitions (10 seeds): 21.6 seconds
  - Bootstrap analysis (B=1,000): 14.8 seconds
  - Publication figures: 7.9 seconds
  - Test suite: 2.7 seconds
  - **Total Reproduction Time**: **~1 minute 18 seconds** on standard CPU.

---

## 6. Pre-Push Action Checklist

Before running `git push -u origin main`:
- [ ] **Action 1**: Sanitize the 80 email addresses in `data/raw/raw_flaky_issues.csv` and `data/raw/github_issues_metadata.json` (replacing emails with `anonymized@example.com`), OR add those two raw scraped files to `.gitignore`.
- [ ] **Action 2**: Maintain `data/raw/DataBase_GitHub.xlsx` and `data/processed/` in Git tracking (both are 100% email-free and essential for immediate cloning and reproduction).
- [ ] **Action 3**: Ensure user approval for the specific GitHub remote URL.
