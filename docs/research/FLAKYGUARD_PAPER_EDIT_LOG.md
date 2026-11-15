# FlakyGuard AI: Section-by-Section Manuscript Edit Log

**Date**: November 2, 2026  
**Document Edited**: [`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)  
**Backup of Original**: [`docs/research/FLAKYGUARD_AI_RESEARCH_PAPER_original_backup.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER_original_backup.md)  

---

### Section 1: Title
- **Change**: Updated title to *"FlakyGuard AI: An Empirical Evaluation of Repository-Level and Cross-Language Generalization, Uncertainty Estimation, and Explainability Limitations for Flaky Test Issue Detection"*.
- **Rationale**: Reflects an empirical evaluation focus, avoiding unjustified claims of novel algorithm design.

### Section 2: Abstract
- **Change**:
  - Replaced "10 repeated `GroupShuffleSplit` partitions with strictly zero repository overlap" with accurate terminology clarifying that partitions are repeated random grouped samples, not independent folds.
  - Specified exact model-specific cluster bootstrap standard error changes (+13% for RF, +4% for SVM, -2% for LR), noting that naive IID bootstrap does not universally underestimate uncertainty.
  - Corrected Leave-One-Language-Out descriptions to distinguish mean Macro-F1 across language test sets (0.6053 for LR) from pooled Macro-F1 across all 1,101 predictions (0.6588 for RF).
  - Clarified that Random Forest MDI is strictly non-negative ($\ge 0$) and does not provide directional polarity.
  - Replaced uniform accuracy baseline comparisons ($1/K$) with empirical Macro-F1 baselines (0.0655 for 13 classes, 0.2383 for 4 classes) and explained that 4-class metric gains are largely structural.

### Section 3: Keywords
- **Change**: Added "Repository-Grouped Partitioning" in place of ambiguous terms.

### Section 4: Introduction
- **Change**: Softened claims regarding ML flakiness triage and added citations to early ML flakiness studies (King et al., 2018; Haben et al., 2021).

### Section 5: Problem Statement
- **Change**: Clarified that Task B evaluates both 13-class and 4-class taxonomies against metric-appropriate empirical baselines.

### Section 6: Research Motivation
- **Change**: Refined the framing to emphasize out-of-repository deployment and cross-language transfer.

### Section 7: Research Gap and Scope of Contribution
- **Change**: Explicitly added an audited "Clarification of Contribution" statement stating that the project is an empirical evaluation, not a novel algorithmic proposal.

### Section 8: Research Questions
- **Change**:
  - RQ2: Rephrased to emphasize stability across repeated repository-grouped random partitions and cluster-robust bootstrapping.
  - RQ4: Rephrased to focus on comparing linear coefficients, tree MDI, and local surrogate methods.
  - RQ5: Rephrased to compare 13-class vs. 4-class performance against metric-appropriate empirical baselines.

### Section 9: Related Work
- **Change**:
  - Added complete discussion and text citations for King et al. (2018) and Haben et al. (2021).
  - Integrated text citations for Gyori et al. (2015) and Zhang et al. (2014) regarding test order dependencies.

### Section 10: Dataset Description and Reconciliation
- **Change**:
  - Maintained reconciled sample counts (1,543 raw, 1,513 deduplicated, 1,101 binary, 589 root cause).
  - Added explicit callout clarifying the origin and distinct scope of the "421 repositories" figure.

### Section 11 & 12: Architecture & Feature Engineering
- **Change**: Clarified training-isolation rules for TF-IDF feature extraction.

### Section 13 & 14: Machine Learning Models & Explainable AI
- **Change**: Formalized the mathematical distinction between linear signed coefficients $w_j \in (-\infty, +\infty)$, non-negative tree MDI $\text{MDI}(j) \ge 0$, and local Shapley values $\phi_j(x)$.

### Section 15: Experimental Design
- **Change**: Explicitly documented that repeated `GroupShuffleSplit` evaluations represent repeated repository-grouped partitions with zero repository overlap *within each partition*, not 10 independent folds of conventional cross-validation.

### Section 16: Evaluation Metrics
- **Change**: Clarified definitions of Macro-F1 vs. Accuracy.

### Section 17: Results: In-Distribution Binary Detection (RQ1)
- **Change**: Verified all metric values against `artifacts/metrics/confidence_interval_report.md`.

### Section 18: Results: Repository-Independent Generalization (RQ2)
- **Change**:
  - Replaced all "independent folds" terminology with "repeated repository-grouped random partitions".
  - Updated Table 4 and text to report model-specific standard error ratios (1.13x for RF, 1.04x for SVM, 0.98x for LR).
  - Clarified that naive IID bootstrap does not universally underestimate uncertainty across all models.

### Section 19: Results: Cross-Language Generalization (RQ3)
- **Change**:
  - Explicitly separated Mean Macro-F1 across language-specific test sets (LR: 0.6053, RF: 0.6002, SVM: 0.4822) from Pooled Macro-F1 across all 1,101 predictions (RF: 0.6588, LR: 0.6539, SVM: 0.5024).
  - Qualified language-specific observations regarding Java and Linear SVM as unisolated hypotheses.

### Section 20: Results: Explainability Analysis (RQ4)
- **Change**:
  - Clarified that the coefficient comparison applies strictly to Logistic Regression on ten curated tokens (9/10, or 90% sign consistency).
  - Documented the sign inversion of `goroutine`.
  - Reasserted that Random Forest MDI is strictly non-negative and does not provide directional signs.

### Section 21: Results: Root-Cause Classification Granularity (RQ5)
- **Change**:
  - Completely overhauled Table 7 and Figure 8 to report both Macro-F1 and Accuracy alongside their respective empirical baselines generated from `src/root_cause_baselines.py`.
  - Removed all statements dividing Macro-F1 by uniform accuracy ($1/K$).
  - Clarified that 4-class performance rise is largely structural (baseline elevation and error absorption), rather than inherent model improvement.

### Section 22 & 23: Limitations & Threats to Validity
- **Change**: Expanded limitations to cover lexical bag-of-words assumptions, minority class scarcity, and the scope of explainability experiments.

### Section 24 & 25: Conclusion & Future Work
- **Change**: Aligned conclusions with empirical findings and toned down speculative generalizations.

### Section 26: References
- **Change**:
  - Reordered and verified all 15 bibliographic references alphabetically.
  - Added complete entries for King et al. (2018) and Haben et al. (2021).
  - Confirmed 100% two-way correspondence between in-text citations and reference list entries.
