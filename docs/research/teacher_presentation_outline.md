# FlakyGuard AI: Faculty Presentation Outline & Slide Deck Defense

**Title**: Explainable Machine Learning for Flaky Test Detection Across Repositories and Programming Languages  
**Student Presenter**: Undergraduate AI/ML Research Team  
**Supervisor / Faculty Evaluator**: Research Review Committee  
**Target Duration**: 15–20 minutes + Q&A  
**Accompanying Paper**: [`FLAKYGUARD_AI_RESEARCH_PAPER.md`](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/docs/research/FLAKYGUARD_AI_RESEARCH_PAPER.md)  

---

## Slide 1: Title & Research Motivation

### Slide Title
**FlakyGuard AI: Generalization Bounds & Explainability for Flaky Test Issue Detection**

### Key Bullet Points
- **The Problem**: Flaky software tests pass and fail non-deterministically without code changes, disrupting CI/CD pipelines and burning developer hours.
- **The Core Question**: Can machine learning detect flaky test reports from developer issue trackers across unseen software repositories and programming languages?
- **Research Scope**: Grounded in the benchmark dataset of Barbosa et al. (*IEEE Transactions on Software Engineering*, 2023).
- **Core Investigation**: Evaluating repository-independent generalization, cross-language transfer, and explainability stability under real-world distribution shift.

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/system_architecture.png` (High-level overview showing issue ingestion to triage).

### Speaker Notes
> *"Good morning, Professor. Today I am presenting FlakyGuard AI, an empirical machine learning research project focused on automated flaky test detection and root-cause classification from developer issue reports. Flaky tests are a massive pain point in modern software engineering—Google and Microsoft report that thousands of engineering hours are wasted weekly debugging tests that fail due to timing, concurrency, or environmental glitches rather than actual regressions. While previous work showed that machine learning models can classify flaky issues, our research addresses a deeper, critical question: Do these models actually generalize when deployed to previously unseen software repositories and new programming languages, or are they just memorizing project-specific terminology? Over the next 15 minutes, I will walk you through our empirical evaluation, methodology corrections, and findings."*

---

## Slide 2: Problem Statement

### Slide Title
**Triage Complexity in Continuous Integration**

### Key Bullet Points
- **Dynamic Reruns are Costly**: Running tests 50–100 times in CI/CD burns immense compute, increases cloud costs, and delays PR merges.
- **Rich Natural Language in Issues**: Developers thoroughly describe flaky symptoms in GitHub issues and PRs (e.g., *"intermittent timeout in CI"*, *"goroutine race on shutdown"*).
- **Three-Fold Research Objective**:
  1. *Task A (Binary Detection)*: Predict whether an issue represents a genuine flaky test ($T$) or a false alarm ($F$).
  2. *Task B (Root-Cause Triage)*: Categorize the underlying technical failure mode (13 fine-grained classes vs. 4 macro-categories).
  3. *Task C (Explainability Fidelity)*: Verify that feature attributions highlight stable domain concepts rather than noise.

### Recommended Visual
- Example code snippet / issue excerpt showing developer discussion vs. non-flaky maintenance issues.

### Speaker Notes
> *"To frame the problem: dynamic detection tools like DeFlaker rerun tests repeatedly to detect non-determinism. While effective, dynamic execution cannot triage issues before tests run, and rerunning large test suites is computationally prohibitive. On the other hand, GitHub issue trackers contain extensive textual context written by developers. Our goal is to determine if classical ML models can accurately classify these reports upon submission. Specifically, we frame this as binary detection—distinguishing genuine flakiness from false alarms—and multi-class root-cause classification, paired with explainability so engineers understand why a test was flagged."*

---

## Slide 3: Research Gap & Novelty

### Slide Title
**Moving Beyond Optimistic In-Distribution Evaluations**

### Key Bullet Points
- **What Prior Work Did (Barbosa et al., 2023)**:
  - Curated GitHub issues across C, Go, Java, JavaScript, and Python.
  - Demonstrated that issue text contains distinctive vocabulary for flaky tests.
- **What Prior Work Overlooked**:
  - *Repository Leakage*: Standard random 80/20 splits allow multiple issues from the same repository in both train and test sets, artificially inflating accuracy.
  - *Post-Hoc Model Selection Bias*: Selecting the 'best architecture' after observing held-out test performance violates evaluation integrity.
  - *Cluster Correlation*: Ignoring repository clustering artificially compresses confidence intervals by treating issues as independent.
- **Our Contribution**:
  - We do *not* claim a novel proprietary algorithm; we conduct a rigorous **second-level empirical research audit**.
  - Repeated repository-independent splits (10 seeds, zero overlap).
  - Unbiased Leave-One-Language-Out evaluation.
  - Cluster bootstrap uncertainty estimation ($B=1,000$).
  - Mathematical deconstruction of root-cause aggregation gains.

### Recommended Visual
- Diagram showing Random Split (issues intermingled across repos) vs. Grouped Split (repositories strictly isolated).

### Speaker Notes
> *"It is important to clearly articulate our research gap and novelty. We do not claim to invent a brand new neural network architecture. Instead, our novelty lies in conducting the first rigorous, second-level generalization and integrity audit of this research domain. In standard software engineering evaluations, researchers often take an entire dataset and perform a simple random 80/20 split. But repositories like Docker or Kubernetes have dozens of issues in the dataset. If Kubernetes issues appear in both training and testing, the model can simply memorize Kubernetes-specific keywords. In our work, we isolate repositories completely across 10 random seeds, eliminate post-hoc model selection bias in cross-language transfer, and account for repository cluster correlation in our statistical testing."*

---

## Slide 4: Dataset & Preprocessing Integrity

### Slide Title
**Empirical Dataset Schema & Count Reconciliation**

### Key Bullet Points
- **Dataset Source**: Barbosa et al. (IEEE TSE 2023) open benchmark.
- **Count Reconciliation**:
  - Resolved conflicting counts: 421 (projects with confirmed flaky issues in author sheets) vs. **641 canonical repositories** across our binary dataset ($N=1,101$).
- **Strict Leakage Prevention**:
  - Combined text: `issue_title + " " + issue_description`.
  - Removed URLs, commit hashes, issue numbers, and reviewer resolution tags.
  - Sublinear TF-IDF vectorizers fitted **exclusively on training splits** in all experiments.
- **Confirmed Label Populations**:
  - *Binary Dataset ($N=1,101$)*: 701 Flaky (63.7%), 400 False Positive (36.3%).
  - *Root-Cause Dataset ($N=589$)*: 13 fine-grained categories across 349 repositories.

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/dataset_distribution.png` (Multi-panel plot: binary split, language breakdown, top root causes).

### Speaker Notes
> *"Here we look at the dataset distribution and our data reconciliation phase. During our initial audit, we noticed an apparent discrepancy: earlier reports cited 421 repositories, while our grouped split partitioned 512 train and 129 test repositories, summing to 641. By digging into the raw Excel spreadsheets from Barbosa et al., we discovered that 421 was explicitly defined by the authors as the number of projects containing at least one confirmed flaky issue, whereas 760 projects were analyzed in total. When we canonicalize owner and repo slugs from GitHub URLs across our 1,101 binary issues, we arrive at exactly 641 unique repositories. We also enforced strict preprocessing isolation: TF-IDF vectorizers are never fitted on the whole dataset—they are fitted strictly on training partitions to prevent data leakage."*

---

## Slide 5: Proposed Framework & Modeling Pipeline

### Slide Title
**Multi-Regime Experimental Testbed**

### Key Bullet Points
- **Feature Space**: Sublinear TF-IDF unigrams & bigrams (max 8,000 features, $\text{min\_df}=2$).
- **Classical Model Suite**:
  - *Random Forest*: Non-linear ensemble (150 trees, max depth 30, balanced class weights).
  - *Logistic Regression*: Regularized linear model ($C=1.0$, balanced weights, signed coefficients).
  - *Linear SVM*: Platt-calibrated linear support vector classifier.
- **Four Rigorous Evaluation Regimes**:
  1. *Stratified Random Holdout (80/20)*: In-distribution baseline ($N=221$).
  2. *Repeated Repository-Independent Evaluation*: 10 random seeds, strictly zero overlap.
  3. *Leave-One-Language-Out (LOLO)*: 5 held-out languages $\times$ 3 models ($5 \times 3$ unbiased matrix).
  4. *Repository-Level Cluster Bootstrap*: $B=1,000$ resamples of repository clusters.

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/system_architecture.png` (Focus on Section 4: Modeling & Regimes).

### Speaker Notes
> *"Slide 5 outlines our modeling framework. We intentionally chose classical machine learning over opaque deep learning models for three reasons: first, classical models are computationally efficient and runnable locally in standard CI/CD runners without GPUs; second, TF-IDF with linear models and tree ensembles provides strong baselines on technical software text; and third, they offer direct mathematical interpretability. We evaluate three distinct architectures—Random Forest, Logistic Regression, and Calibrated Linear SVM—across four separate experimental regimes, including repeated grouped splits and cluster-robust bootstrapping."*

---

## Slide 6: In-Distribution Binary Classification (RQ1)

### Slide Title
**Binary Flakiness Detection Performance**

### Key Bullet Points
- **Research Question 1**: How effectively can models detect flaky issues under standard in-distribution conditions?
- **Random Holdout Results ($N=221$, Test Flaky Prior: 63.3%)**:
  - **Random Forest**: **Macro-F1: 0.6848** (95% CI: [0.6235, 0.7466]), Accuracy: 72.4%, ROC-AUC: 0.7310, PR-AUC: 0.8258.
  - **Logistic Regression**: **Macro-F1: 0.6555** (95% CI: [0.5946, 0.7198]), Accuracy: 69.2%, ROC-AUC: 0.7262, PR-AUC: 0.8066.
  - **Linear SVM**: **Macro-F1: 0.6107** (95% CI: [0.5400, 0.6757]), Accuracy: 68.8%, ROC-AUC: 0.6993, PR-AUC: 0.8035.
- **Baseline Comparison**: All candidate models significantly exceed the majority prior baseline (Macro-F1: **0.4550**).

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/binary_model_comparison.png` (Bar chart with metrics and prior baseline).

### Speaker Notes
> *"Slide 6 presents our in-distribution binary classification results for RQ1. On the 20% holdout test set of 221 issues, Random Forest emerges as the strongest in-distribution classifier, achieving a Macro-F1 of 0.6848, an Accuracy of 72.4%, and a PR-AUC of 0.8258. Notice that under empirical class imbalance—where 63.3% of test issues are flaky—a naive majority-class classifier achieves a Macro-F1 of only 0.455. Both Random Forest and Logistic Regression surpass this chance baseline by more than 20 percentage points. Bootstrapping with 1,000 iterations establishes an empirical 95% confidence interval of [0.6235, 0.7466] for Random Forest."*

---

## Slide 7: Generalization Under Distribution Shift (RQ2 & RQ3)

### Slide Title
**Cross-Repository & Cross-Language Robustness**

### Key Bullet Points
- **RQ2: Repository-Independent Generalization (10 Partitions)**:
  - Evaluated across 10 repeated repository-grouped random partitions (`GroupShuffleSplit`) with strictly 0 overlapping repositories within each partition.
  - **Logistic Regression**: Macro-F1 = **0.6670 $\pm$ 0.0239**; ROC-AUC = **0.7265 $\pm$ 0.0380**.
  - **Random Forest**: Macro-F1 = **0.6542 $\pm$ 0.0267**; ROC-AUC = **0.7091 $\pm$ 0.0346**.
  - *Catastrophic collapse hypothesis is falsified*: Performance variation across partitions is modest ($\text{Std} \le 0.027$).
  - *Cluster Bootstrap Impact*: Standard error inflates by **$1.13\times$** for Random Forest (0.0349 vs. 0.0310) and $1.04\times$ for Linear SVM, while remaining virtually unchanged for Logistic Regression (0.0303 vs. 0.0308; $0.98\times$).
- **RQ3: Cross-Language Generalization (Unbiased LOLO Matrix)**:
  - Evaluated on held-out ecosystems without post-hoc selection:
  - **Logistic Regression**: Mean Macro-F1 across languages = **0.6053 $\pm$ 0.0818** (Top performing).
  - **Random Forest**: Mean Macro-F1 across languages = **0.6002 $\pm$ 0.0722**.
  - **Linear SVM**: Degrades to **0.4822 $\pm$ 0.0610** due to cross-language feature shifts.
  - Ecosystem variation: Transfer is highest in JavaScript (0.69) and C (0.66), and lower in Java (0.51).
  - *Pooled Cross-Language Predictions ($N=1,101$)*: RF = 0.6588; LR = 0.6539; SVM = 0.5024.

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/repository_independent_comparison.png` AND `artifacts/figures/publication/lolo_language_comparison.png`.

### Speaker Notes
> *"Now we move to the core research findings in Slide 7: cross-repository and cross-language generalization. For RQ2, we tested whether performance collapses when the model is tested strictly on repositories it has never seen before. Across 10 repeated repository-grouped random partitions with zero repository overlap, Logistic Regression achieved a mean Macro-F1 of 0.6670, and Random Forest achieved 0.6542 across partitions. The standard deviation across repetitions was only 0.024, proving that our models do not rely on repository-specific overfitting. However, our cluster bootstrap showed that treating issues as independent underestimates variance by 13% for Random Forest, though the clustering effect is model-specific. For RQ3, our unbiased LOLO evaluation revealed that Logistic Regression is the most robust cross-language architecture, achieving a mean Macro-F1 of 0.6053 across the 5 language test sets. Linear SVM degrades when vocabulary shifts, scoring 0.4822. Furthermore, Java proved the most difficult held-out language, possibly reflecting framework-specific idioms."*

---

## 8. Root-Cause Classification & Explainability (RQ4 & RQ5)

### Slide Title
**Root-Cause Diagnosis & Explainability Audit**

### Key Bullet Points
- **RQ5: Root-Cause Classification Granularity**:
  - *13 Fine-Grained Classes ($N=118$ test)*: Random Forest achieves Macro-F1 of **0.2834** (95% CI: [0.1980, 0.3494]) and Accuracy of **38.14%** (vs. uniform-random Macro-F1: 0.0655; uniform Accuracy: 7.76%).
  - *4 Macroscopic Categories*: Logistic Regression achieves Macro-F1 of **0.4222** and Accuracy of **54.24%** (vs. uniform-random Macro-F1: 0.2383; uniform Accuracy: 25.07%).
  - *Deconstruction of the Rise*: The rise is driven by chance baseline elevation and within-domain error absorption, **not** inherent algorithmic improvement.
- **RQ4: Explainability Stability & MDI Mechanics**:
  - *SHAP Global Analysis*: Core indicators (`flaky`, `timeout`, `race`, `intermittent`) consistently drive flakiness predictions.
  - *Sign Invariance Audit*: 9 out of 10 curated tokens maintain identical sign across repository shift (90%, not 100%). Language-specific keyword `goroutine` inverted sign ($-0.17 \rightarrow +0.04$).
  - *Mathematical Clarification*: Random Forest MDI Gini importance is **strictly non-negative ($\ge 0$)**; signed attributions require SHAP or LIME.

### Recommended Visual
- **Artifact**: `artifacts/figures/publication/root_cause_13_vs_4_comparison.png` AND `artifacts/figures/publication/explainability_stability.png`.

### Speaker Notes
> *"Slide 8 covers root-cause diagnosis and our explainability audit. In Task B, classifying confirmed flaky issues into 13 fine-grained categories yielded a Macro-F1 of 0.2834. While seemingly modest, our empirical baseline analysis shows that a uniform-random classifier achieves a Macro-F1 of only 0.0655—meaning our model achieves 4.3 times the chance Macro-F1. When we aggregate the 13 classes into 4 high-level categories—Timing, Concurrency, Environment, and Test Logic—Macro-F1 rises to 0.4222. But as scientists, we must separate structural effects from model learning: the uniform Macro-F1 baseline jumps from 0.0655 to 0.2383, and within-domain confusions are absorbed as true positives. In our explainability audit, we dissected the earlier claim of '100% sign invariance'. We showed that while 9 of 10 keywords remained stable in Logistic Regression, 'goroutine' inverted signs due to uneven Go repository clustering. We also clarified that Random Forest Gini importances cannot have signs—signed directional explanations require SHAP Shapley values."*


---

## Slide 9: Limitations & Research Contributions

### Slide Title
**Threats to Validity & Core Contributions**

### Key Bullet Points
- **Key Limitations**:
  - *Issue Text Proxy*: Analyzes human developer descriptions, not runtime dynamic execution traces or thread schedules.
  - *Minority Sample Scarcity*: Extreme rarity in specific root-cause categories (e.g., `Test Suite Timeout`, $N=5$).
  - *Linguistic Bag-of-Words*: TF-IDF captures lexical cues but omits deep sequential syntax and code semantics.
- **Summary of Contributions**:
  1. *First Multi-Seed Repository Audit*: Proved stability across 10 random grouped splits with 0 overlap.
  2. *Unbiased LOLO Baseline*: Established true cross-language transfer without post-hoc selection.
  3. *Cluster-Robust Inference*: Demonstrated the necessity of cluster bootstrapping for SE repository data.
  4. *Deconstructed Metric Gains*: Provided mathematical rigor separating true model gains from baseline shifts.
  5. *Full Open Reproduction Package*: 25 passing automated tests, runnable local pipeline.

### Recommended Visual
- Summary table distinguishing Confirmatory Benchmarks vs. Exploratory Operational Findings.

### Speaker Notes
> *"On Slide 9, we address the limitations and formal contributions of our work. The primary limitation is that our model classifies issue reports—it sees what developers write, not runtime JVM thread dumps. Additionally, rare root-cause categories suffer from small sample sizes. However, our contributions to the academic literature are substantial: we implemented the first multi-seed repository-independent audit with zero overlap, established an unbiased cross-language evaluation matrix, proved the impact of cluster correlation on confidence intervals, and dismantled common evaluation artifacts in category aggregation and feature explainability. Everything is verified with 25 passing automated unit tests."*

---

## Slide 10: Conclusion & Future Work

### Slide Title
**Summary & Future Research Directions**

### Key Bullet Points
- **Final Verdict**: Classical ML models (Logistic Regression & Random Forest) generalize reliably across software repositories (Macro-F1: ~0.66) and transfer across languages (Macro-F1: ~0.60), confirming that developers share a common vocabulary of non-determinism.
- **Practical Recommendation**:
  - Deploy **Logistic Regression** for fast, calibrated, explainable cross-language triage.
  - Deploy a **Two-Tier Root-Cause System**: 4-class router for team dispatch, 13-class top-3 ranking for developer debugging.
- **Future Work**:
  - Fine-tuning code-aware foundation models (CodeBERT / StarCoder) on issue text.
  - Multi-modal fusion combining issue text with CI/CD runtime metrics (test execution time, churn).
  - Active learning to gather targeted annotations on rare root-cause failure modes.

### Recommended Visual
- Closing summary slide with repository URL and contact information.

### Speaker Notes
> *"To conclude: FlakyGuard AI proves that classical natural language models can reliably detect flaky test issues without collapsing under repository or language distribution shift. For production CI/CD deployments, we recommend Logistic Regression for its calibrated probabilities and cross-language robustness, coupled with a two-tier triage approach for root causes. Looking ahead, the natural next step is fine-tuning pre-trained code language models like CodeBERT, and fusing issue text with runtime execution logs. Thank you for your time and guidance, Professor. I am now ready for your questions."*

---

## Appendix: Anticipated Faculty Q&A Preparation

| Anticipated Supervisor Question | Concise Methodological Answer | Reference in Paper |
| :--- | :--- | :--- |
| **Q1: Why did you choose classical ML instead of fine-tuning BERT or LLMs?** | Classical ML (TF-IDF + LR/RF) is computationally lightweight, deterministic, requires no GPU infrastructure, and serves as the mandatory rigorous baseline before deep learning complexity is justified. | Section 13 & 14 |
| **Q2: Why did Linear SVM collapse in cross-language evaluation (LOLO)?** | Linear SVM optimizes an uncalibrated margin boundary. When evaluated on an unseen language whose specific vocabulary was absent during training, feature sparsity causes the decision score to drift, collapsing all predictions to the majority class. | Section 19 |
| **Q3: What is the practical difference between cluster bootstrap and regular bootstrap?** | Regular bootstrap resamples individual issues assuming i.i.d. observations. Cluster bootstrap resamples entire repositories with replacement. Because issues from the same repository share latent correlations, cluster bootstrap correctly widens standard errors by 10–15%. | Section 15 & 18 |
| **Q4: Why does 4-class root cause classification have a higher Macro-F1?** | As proven in Section 21, reducing classes from 13 to 4 shifts the uniform chance baseline from 7.7% to 25%, and eliminates false-negative penalties between confusable subcategories like Async Wait and Timeout. | Section 21 |
| **Q5: Why did `goroutine` invert sign between random split and repo-independent split?** | In the random split, several non-flaky Go issues containing `goroutine` pulled the weight negative. In the repo-independent split, those specific Go projects were placed into the test set, leaving training Go issues with genuine race conditions, pulling the weight positive. | Section 20 |
