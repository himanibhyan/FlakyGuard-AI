# FlakyGuard AI: Final Faculty-Review-Level Methodology Audit & Evaluation Report

**Date**: November 2, 2026  
**Reviewer Role**: Independent Academic Reviewer, Empirical Software Engineering Researcher, & Methodology Auditor  
**Evaluated Manuscript**: [`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)  
**Accompanying Presentation**: [`docs/research/teacher_presentation_outline.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/teacher_presentation_outline.md)  
**Audit Scope**: Independent verification of underlying data, scripts, metric calculations, baseline methodologies, and claims.  

---

## 1. Overall Assessment

This audit independently inspected the entire FlakyGuard AI experimental codebase, scripts, serialized models, metrics CSVs, figures, and bibliographic citations. The assessment evaluates the readiness of the revised research manuscript across three academic milestones:

1. **Readiness for Faculty Supervisor Discussion**: **HIGH**. The manuscript is methodologically sound, candid about structural baseline mechanisms, and backed by verifiable experimental artifacts. Every claim connects to a reproducible script and CSV.
2. **Readiness for Research Presentation / Defense**: **HIGH**. The oral presentation deck and manuscript now use strictly accurate terminology (e.g., repeated repository-grouped partitions rather than cross-validation folds) and present metric-appropriate baseline comparisons.
3. **Readiness for Formal Academic Archival / Submission**: **MODERATE-TO-HIGH**. The manuscript meets standards for workshop/conference submission regarding classical ML baselines and generalization audits in empirical software engineering. However, peer reviewers in premier venues (e.g., ICSE, FSE, TSE) will expect future extensions comparing classical models against fine-tuned code-aware foundation models (e.g., CodeBERT) and multi-modal signals.

> [!IMPORTANT]
> **Scientific Integrity Notice**: No empirical study on observational issue data can provide an absolute guarantee of causal test failure diagnosis. The paper is scientifically defensible because it explicitly scopes its findings to developer-written natural language reports, refrains from asserting causal mechanisms, and acknowledges the limitations of post-hoc feature explanations.

---

## 2. Research Question Alignment

The table below audits the alignment between each research question, its supporting experiment, the empirical evidence, its boundary limitations, and the reviewer verdict:

| Research Question | Supporting Experiment | Empirical Evidence | Boundary Limitations | Reviewer Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **RQ1 (In-Distribution Binary Detection)** | Stratified 80/20 train/test holdout ($N=880$ train, $N=221$ test), TF-IDF unigram+bigram. | Random Forest: Macro-F1 = **0.6848** (95% CI: [0.6235, 0.7466]), Acc = **72.40%**, ROC-AUC = **0.7310**, PR-AUC = **0.8258** vs. Majority prior baseline Macro-F1 = **0.4550**. | Evaluated on holdout data where 35.3% of repositories overlap between train and test; in-distribution only. | **Supported** |
| **RQ2 (Repository-Independent Generalization)** | 10 repeated `GroupShuffleSplit` partitions with fixed seeds (`[42..909]`); cluster bootstrap ($B=1,000$). | LR Mean Macro-F1 = **0.6670 $\pm$ 0.0239**; RF = **0.6542 $\pm$ 0.0267** across partitions. Cluster bootstrap SE inflation is model-specific: RF = $1.13\times$ (0.0349 vs 0.0310); SVM = $1.04\times$; LR = $0.98\times$. | Partitions are repeated random samples, not independent folds. Cluster bootstrap effect was evaluated on a single test partition (Seed 42). | **Supported** |
| **RQ3 (Cross-Language Generalization)** | Leave-One-Language-Out (LOLO) across C, Go, Java, JS, Python ($5 \times 3$ unbiased matrix). | LR Mean Macro-F1 across language test sets = **0.6053 $\pm$ 0.0818**; RF = **0.6002 $\pm$ 0.0722**; SVM = **0.4822 $\pm$ 0.0610**. Pooled LOLO ($N=1,101$): RF = **0.6588**; LR = **0.6539**. Java transfer is lowest (0.5144); JS highest (0.6929). | Language boundaries correlate with project domains and issue styles; experiments do not isolate syntax from community writing conventions. | **Supported** |
| **RQ4 (Explainability Mechanics & Consistency)** | Comparison of LR linear coefficients on 10 curated domain tokens across repository shift; global SHAP analysis. | 9/10 tokens (90%) maintained sign; `goroutine` inverted sign ($-0.1708 \rightarrow +0.0387$). RF MDI Gini importances are strictly non-negative ($\ge 0$) and cannot indicate direction. | Evaluated on ten curated keywords under Logistic Regression; does not evaluate full-vocabulary stability or prove causal faithfulness. | **Supported (Properly Scoped)** |
| **RQ5 (Root-Cause Taxonomy Granularity)** | Holdout evaluation on confirmed flaky reports ($N=589$, test $N=118$) across 13 classes vs. 4 macro categories with Monte Carlo baselines. | 13-class RF: Macro-F1 = **0.2834**, Acc = **38.14%** (vs. Uniform Macro-F1: 0.0655; Uniform Acc: 7.76%). 4-class LR: Macro-F1 = **0.4222**, Acc = **54.24%** (vs. Uniform Macro-F1: 0.2383; Uniform Acc: 25.07%). | Metric rise reflects baseline elevation and within-domain error absorption alongside sample pooling, rather than an inherent algorithmic breakthrough. | **Supported** |

---

## 3. Statistical and Methodological Audit

### 3.1 Repository Separation
- **Verification**: Independently verified `src/repository_generalization.py` and `src/repeated_repository_evaluation.py`.
- **Findings**: The regex extractor `parse_canonical_repo` extracts `{owner}/{repo}` from `url_issue`. In all 10 evaluated partitions, the intersection between training repositories and testing repositories is strictly empty:
  $$\text{Train Repositories} \cap \text{Test Repositories} = \emptyset$$
  Total unique repositories across the binary dataset ($N=1,101$) is verified at **641**.

### 3.2 Leave-One-Language-Out (LOLO)
- **Verification**: Checked `src/lolo_evaluation.py` and `artifacts/metrics/lolo_model_selection_audit.md`.
- **Findings**: For each held-out language, vectorization vocabulary and IDF statistics are fitted strictly on the four remaining languages. No held-out language samples leak into vectorizer construction or model tuning. The manuscript correctly reports the full $5 \times 3$ unbiased matrix and separates language-mean Macro-F1 from pooled Macro-F1.

### 3.3 Repeated Grouped Partitions vs. Cross-Validation
- **Verification**: Audited all textual descriptions of repeated `GroupShuffleSplit` runs.
- **Findings**: The paper accurately avoids the inaccurate phrases "10-fold cross validation" and "10 independent folds". It correctly describes them as **10 repeated repository-grouped random partitions**, clarifying that while zero repository overlap is enforced within each partition, test repositories may overlap across different repetitions.

### 3.4 Bootstrap and Uncertainty Estimation
- **Verification**: Checked `src/confidence_intervals.py` and `artifacts/metrics/confidence_interval_report.md`.
- **Findings**: Non-parametric bootstrap ($B=1,000$, seed 42) was verified. The cluster bootstrap resamples entire repository clusters with replacement. The manuscript now accurately reports that the standard error adjustment is model-specific:
  - Random Forest: Cluster SE = 0.0349 vs. IID SE = 0.0310 ($1.13\times$)
  - Linear SVM: Cluster SE = 0.0355 vs. IID SE = 0.0341 ($1.04\times$)
  - Logistic Regression: Cluster SE = 0.0303 vs. IID SE = 0.0308 ($0.98\times$)
  The previously exaggerated claim that naive IID bootstrap "always underestimates uncertainty by 30–40%" has been excised.

### 3.5 Root-Cause Empirical Baselines
- **Verification**: Checked `src/root_cause_baselines.py` and `artifacts/metrics/root_cause_baselines.csv`.
- **Findings**: Evaluated on the exact test set ($N=118$, with all 13 classes represented in the test set ground truth):
  - *Majority Class Baseline*: Training set majority for 13 classes is `Async Wait` (class 0). In the test set, class 0 has 21 samples. Accuracy is $21/118 = 17.80\%$, Macro-F1 is $0.3022 / 13 = 0.0232$. For 4 classes, training majority is `Resource & Environment` (44 test samples). Accuracy is $44/118 = 37.29\%$, Macro-F1 is $0.5432 / 4 = 0.1358$.
  - *Uniform-Random Baseline (Expected over 1,000 trials, seed 42)*: 13-class Macro-F1 = **0.0655**, Accuracy = **7.76%** (approx $1/13 = 7.69\%$). 4-class Macro-F1 = **0.2383**, Accuracy = **25.07%** (approx $1/4 = 25.00\%$).
  - *Metric Separation*: The previous error of dividing Macro-F1 by an Accuracy baseline ($1/K$) was completely eliminated. The manuscript and Figure 8 now clearly distinguish Macro-F1 baselines from Accuracy baselines.

---

## 4. Explainability Audit

### What the Existing Evidence Supports:
1. **Global Linear Attributions**: In Logistic Regression, global coefficients identify intuitive software engineering terms (`flaky`, `timeout`, `race`, `intermittent`, `wait`, `sleep`) as positive predictors of flakiness, while routine maintenance terms (`add`, `update`, `version`, `support`) correlate with non-flaky reports.
2. **Curated Token Sign Stability**: Comparing Logistic Regression coefficients between random split (with overlap) and repo-independent split (zero overlap) across 10 curated domain tokens yields **90% sign consistency (9/10)**.
3. **Distribution-Induced Polarity Shift**: The Go-specific token `goroutine` inverted sign ($-0.1708 \rightarrow +0.0387$), providing direct empirical proof that repository clustering can alter token polarity for language-specific keywords.
4. **Non-Negativity of Tree MDI**: Random Forest Mean Decrease in Impurity (Gini importance) is mathematically non-negative ($\ge 0$) and cannot indicate directional polarity.

### What the Evidence Does NOT Support (Properly Excised):
1. Does **not** support "100% sign invariance across distribution shift".
2. Does **not** support that Random Forest features have "stable positive signs".
3. Does **not** support "comprehensive explanation robustness" across the entire 8,000-feature vocabulary.
4. Does **not** establish causal faithfulness to runtime JVM/OS execution mechanics.

---

## 5. Citation and Related-Work Audit

All 15 references listed in Section 26 of the revised manuscript were independently verified:

| # | Reference | Venue / Identifier | In-Text Citation Located | Support Validated |
| :---: | :--- | :--- | :---: | :---: |
| 1 | **Allamanis (2019)** | *ACM Onward! 2019* (DOI: 10.1145/3359591.3359735) | Section 4, 9.3 | **Verified** |
| 2 | **Barbosa et al. (2023)** | *IEEE TSE 2023* (DOI: 10.1109/TSE.2022.3208864) | Throughout | **Verified** |
| 3 | **Bell et al. (2018)** | *ACM/IEEE ICSE 2018* (DOI: 10.1145/3180155.3180164) | Section 9.2 | **Verified** |
| 4 | **Cameron & Miller (2015)** | *J. Human Resources 2015* (DOI: 10.3368/jhr.50.2.317) | Section 9.3 | **Verified** |
| 5 | **Efron & Tibshirani (1994)** | *CRC Press 1994* (Monograph) | Section 9.3 | **Verified** |
| 6 | **Gruber et al. (2021)** | *IEEE/ACM ASE 2021* (DOI: 10.1109/ASE51524.2021.9678783) | Section 9.1 | **Verified** |
| 7 | **Gyori et al. (2015)** | *ACM ESEC/FSE 2015* (DOI: 10.1145/2786805.2786853) | Section 9.1 | **Verified** |
| 8 | **Haben et al. (2021)** | *IEEE/ACM MSR 2021* (DOI: 10.1109/MSR52588.2021.00033) | Section 4, 7, 9.2 | **Verified** |
| 9 | **King et al. (2018)** | *IEEE QRS-C 2018* (DOI: 10.1109/QRS-C.2018.00030) | Section 4, 7, 9.2 | **Verified** |
| 10 | **Lam et al. (2019)** | *IEEE ICST 2019* (DOI: 10.1109/ICST.2019.00024) | Section 9.1 | **Verified** |
| 11 | **Lundberg & Lee (2017)** | *NeurIPS 2017* (Vol. 30) | Section 9.4 | **Verified** |
| 12 | **Luo et al. (2014)** | *ACM FSE 2014* (DOI: 10.1145/2635868.2635920) | Section 4, 9.1 | **Verified** |
| 13 | **Munaiah et al. (2017)** | *Empirical Softw. Eng. 2017* (DOI: 10.1007/s10664-017-9512-6) | Section 4, 9.3 | **Verified** |
| 14 | **Ribeiro et al. (2016)** | *ACM KDD 2016* (DOI: 10.1145/2939672.2939778) | Section 9.4 | **Verified** |
| 15 | **Zhang et al. (2014)** | *ACM ISSTA 2014* (DOI: 10.1145/2610384.2610410) | Section 9.1 | **Verified** |

- **Citation Consistency**: Exactly 15 in-text citations match the 15 reference entries (100% two-way correspondence). No phantom or uncited entries exist.

---

## 6. Numerical Discrepancies & Reconciliation Record

| Reported Metric | Previous Paper Text | Verified Evidence Artifact | Final Reconciled Text | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Repository Counts** | "421 repositories" | Raw Excel tab `ProjectsDomain` row 2 defines 421 as projects with $\ge 1$ flaky issue. Cleaned binary dataset has **641 canonical repositories**. | Clarified distinct scopes: 421 is confirmed flaky projects in author sheets; 641 is canonical repositories in binary dataset. | **Reconciled** |
| **Repeated Split Performance** | "10 independent folds" | `GroupShuffleSplit` across 10 random seeds with fixed test size = 0.20. | Replaced with "10 repeated repository-grouped random partitions". | **Reconciled** |
| **Root-Cause Baselines** | $1/13 = 7.69\%$ and $1/4 = 25.00\%$ presented as Macro-F1 baselines. | Accuracy and Macro-F1 are distinct metrics. Empirical uniform-random Macro-F1 is **0.0655** (13-class) and **0.2383** (4-class). | Separated Macro-F1 from Accuracy baselines in Table 7 and Figure 8. | **Reconciled** |
| **Cluster Bootstrap Inflation** | "Always underestimates uncertainty by 30–40%" | `confidence_interval_report.md` records SE ratios: RF = 1.13, SVM = 1.04, LR = 0.98. | Reported model-specific adjustments (+13% for RF, +4% for SVM, -2% for LR). | **Reconciled** |
| **Sign Consistency** | "100% sign invariance" | `explainability_claim_audit.md` shows `goroutine` inverted sign ($-0.1708 \rightarrow +0.0387$). | Reported 90% (9/10) on curated tokens with Go keyword inversion. | **Reconciled** |

---

## 7. Remaining Limitations (To Candidly Acknowledge During Supervisor Defense)

1. **Human Language as Failure Proxy**: The system triages developer natural language reports submitted after an issue occurs. It does not inspect dynamic runtime thread interleavings or memory access traces.
2. **Lexical Representation**: Unigram and bigram TF-IDF models word frequencies and bigrams effectively, but does not capture deep sequential semantics or syntax in code snippets.
3. **Minority Class Sparsity**: With 589 confirmed root-cause annotations across 13 categories, rare classes (e.g., `Test Suite Timeout`, $N=5$; `Too Restrictive Range`, $N=6$) have limited statistical support for supervised learning.
4. **Explainability Breadth**: The attribution stability experiment was conducted on a curated set of ten keywords in Logistic Regression; broader vocabulary stability under non-linear models warrants future work.

---

## 8. Summary of Manuscript Changes

1. **Title**: Updated to emphasize an empirical evaluation rather than an unvalidated claim of novel architecture.
2. **Abstract**: Structured around the five research questions, incorporating exact verified metrics, corrected repeated-partitions terminology, and empirical baselines.
3. **Section 8 (Research Questions)**: Refined RQ4 to focus on linear model feature coefficients, tree MDI mechanics, and attribution consistency.
4. **Section 9 & 26 (Related Work & References)**: Added complete, verified citations for King et al. (2018) and Haben et al. (2021); integrated text citations for Gyori et al. (2015) and Zhang et al. (2014).
5. **Section 10 (Dataset Description)**: Maintained reconciled sample counts (1,543 raw, 1,513 deduplicated, 1,101 binary, 589 root cause) and clarified the 421 repository scope.
6. **Section 18 (Repository-Independent Generalization)**: Replaced all "independent folds" terminology with "repeated repository-grouped random partitions" and reported model-specific cluster bootstrap ratios.
7. **Section 19 (Cross-Language Generalization)**: Separated Mean Macro-F1 across language-specific test sets from Pooled Macro-F1, and qualified Java/SVM discussions as observational hypotheses.
8. **Section 20 (Explainability Mechanics)**: Explicitly scoped the attribution experiment to ten curated tokens in Logistic Regression, documented `goroutine`'s sign inversion, and formalized the non-negativity of Random Forest MDI.
9. **Section 21 (Root-Cause Granularity)**: Overhauled Table 7 to report both Macro-F1 and Accuracy alongside their respective empirical baselines, and explained that 4-class gains reflect baseline elevation, error absorption, and representation density.
10. **Section 24 (Conclusion)**: Explicitly answered all five research questions individually.
11. **Figure 4, 6, and 8**: Regenerated via `src/publication_figures.py` with corrected terminology ("Partitions") and two-panel separation of Macro-F1 from Accuracy baselines.

---

## 9. Final Review Verdict

### **VERDICT: READY FOR FACULTY DISCUSSION AND RESEARCH PRESENTATION.**

The revised manuscript [`FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md) is methodologically defensible, mathematically grounded, and supported by 25 passing automated tests (`pytest tests/`). It avoids overclaiming, clearly separates empirical findings from observational hypotheses, and candidly discusses baseline artifacts and explainability constraints.
