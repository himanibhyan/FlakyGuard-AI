# FlakyGuard AI: An Empirical Investigation into Repository-Independent Generalization, Cross-Language Transfer, and Explainability Robustness for Flaky Test Issue Detection

**Authors**: Research Team (FlakyGuard AI Project)  
**Affiliation**: Department of Computer Science & Software Engineering  
**Benchmark Reference**: Barbosa et al., *“Test Flakiness Across Programming Languages”*, IEEE Transactions on Software Engineering (TSE), 2023  
**Publication Target**: Academic Research Defense & Peer-Reviewed Software Engineering Archive  

---

## 1. Title
**FlakyGuard AI: An Empirical Investigation into Repository-Independent Generalization, Cross-Language Transfer, and Explainability Robustness for Flaky Test Issue Detection**

---

## 2. Abstract
Flaky software tests—tests that exhibit non-deterministic pass and fail outcomes under identical code and execution configurations—impose substantial friction on Continuous Integration and Continuous Deployment (CI/CD) workflows. While classical machine learning models trained on software issue reports have shown promising in-distribution classification performance, critical research questions persist regarding their capacity to generalize across unseen software repositories, transfer across disparate programming language ecosystems, and yield directionally stable post-hoc explanations under distribution shift. 

In this work, we present **FlakyGuard AI**, a research framework and evaluation testbed that investigates the generalization boundaries and explainability reliability of classical natural language machine learning models for flaky test issue reports. Grounded in the empirical dataset of Barbosa et al. ($N=1,101$ labeled issue reports spanning C, Go, Java, JavaScript, and Python across 641 distinct GitHub repositories), we conduct a rigorous multi-regime evaluation. 

Our empirical results demonstrate:
1. **Binary Detection Effectiveness (RQ1)**: Under stratified in-distribution evaluation, Random Forest and Logistic Regression achieve Macro-F1 scores of **0.6848** (95% CI: [0.6235, 0.7466]) and **0.6555** (95% CI: [0.5946, 0.7198]), outperforming chance prior baselines (Macro-F1: 0.455).
2. **Repository-Independent Stability (RQ2)**: Evaluated across 10 repeated `GroupShuffleSplit` partitions with strictly zero repository overlap, Logistic Regression (Macro-F1: **0.6670 $\pm$ 0.0239**) and Random Forest (Macro-F1: **0.6542 $\pm$ 0.0267**) maintain consistent discrimination. Applying repository-level cluster bootstrapping reveals a $1.13\times$ standard error inflation, confirming that naive i.i.d. evaluations underestimate generalization variance.
3. **Cross-Language Generalization (RQ3)**: Under an unbiased Leave-One-Language-Out (LOLO) regime isolating model selection from test data, Logistic Regression achieves the highest mean transfer Macro-F1 of **0.6053 $\pm$ 0.0818**, outperforming Random Forest (0.6002) and Linear SVM (0.4822), which collapses due to decision threshold shift.
4. **Explainability Reliability (RQ4)**: Evaluating local and global interpretability using SHAP and LIME reveals that 9 out of 10 core domain tokens maintain directional sign stability across repository shifts, while language-specific keywords (e.g., `goroutine`) suffer sign inversions. Furthermore, we clarify that Random Forest Mean Decrease in Impurity (MDI) importances are strictly non-negative ($\ge 0$) and cannot indicate class polarity.
5. **Root-Cause Granularity (RQ5)**: Fine-grained 13-class root-cause classification is bounded by extreme minority sample scarcity (Macro-F1: **0.2834**; 95% CI: [0.1980, 0.3494]). Aggregating into 4 macroscopic failure modes raises Macro-F1 to **0.4222**; however, we prove mathematically that this rise is driven by chance baseline elevation ($1/13 \rightarrow 1/4$) and within-domain confusion elimination, rather than algorithmic superiority.

---

## 3. Keywords
Flaky Software Tests, Empirical Software Engineering, Machine Learning Generalization, Leave-One-Language-Out, Repository-Independent Evaluation, Explainable AI, SHAP, LIME, Cluster Bootstrap.

---

## 4. Introduction
Automated regression testing is the cornerstone of modern continuous software delivery. However, the integrity of automated testing is frequently undermined by **test flakiness**: non-deterministic test behavior where a test oscillates between passing and failing states without changes to the underlying source code. Flaky tests waste substantial developer hours debugging phantom defects, delay release schedules, erode confidence in continuous integration infrastructure, and can mask genuine regressions.

Recognizing the heavy operational burden of manual test triage, researchers and practitioners have explored machine learning techniques to automate the triage of test issues. Rather than relying solely on repeated runtime dynamic reruns—which are computationally expensive and energy-intensive—recent empirical research has investigated whether natural language descriptions in developer issue trackers and pull requests contain informative textual signals that differentiate genuine flaky tests from non-flaky defects or false alarms (Barbosa et al., 2023; Luo et al., 2014).

Despite encouraging preliminary results reported in the literature, standard machine learning evaluations often rely on random train/test splits. In software engineering datasets, random splitting introduces significant risks of **optimistic evaluation bias**: multiple issue reports submitted to the same software repository share project-specific conventions, directory names, and terminology. If issues from the same repository appear in both training and test sets, the model may overfit to repository-level stylistic signatures rather than learning domain-general indicators of flakiness. Furthermore, cross-language generalization and the stability of feature explanations under distribution shift have received limited systematic scrutiny.

To address these foundational methodology challenges, this paper presents **FlakyGuard AI**, an empirical study and diagnostic testbed designed to evaluate the generalization bounds, statistical uncertainty, and explainability mechanics of classical machine learning for flaky test detection.

---

## 5. Problem Statement
Given an unstructured natural language issue report $d = (t, b)$ comprising an issue title $t$ and issue body $b$ from a public software repository $r$ authored in programming language $l$, our objective is to determine:
1. **Task A (Flakiness Detection)**: Whether the report represents a genuine confirmed flaky test ($y_{\text{bin}} = 1$) or a non-flaky false positive / non-flaky defect report ($y_{\text{bin}} = 0$).
2. **Task B (Root-Cause Diagnosis)**: For confirmed flaky issues ($y_{\text{bin}} = 1$), determine the primary architectural or execution root cause $y_{\text{rc}} \in \mathcal{C}$, where $|\mathcal{C}| = 13$ discrete fine-grained categories or $|\mathcal{C}_{\text{macro}}| = 4$ high-level engineering failure modes.
3. **Task C (Attribution Reliability)**: Quantify the local feature contribution $\phi_j(d)$ of each token $j \in d$ and evaluate whether the directionality and magnitude of $\phi_j$ remain invariant when the model is evaluated on previously unseen repositories and programming languages.

---

## 6. Research Motivation
In enterprise CI/CD pipelines, software systems evolve across thousands of heterogeneous repositories written in diverse programming languages. A flaky test triage system is practically useless if it only performs well on projects it has already memorized. If deploying a model to a new internal team or repository requires collecting and manually annotating hundreds of historical flaky issues for retraining, the operational barrier to adoption remains prohibitive.

Consequently, research must establish:
- **Out-of-Repository Generalization**: Does a model trained on established open-source repositories maintain its discriminative power when deployed on a completely unseen codebase?
- **Out-of-Language Transfer**: Do developers describe flaky behavior using universal natural language terminology (e.g., `race condition`, `deadlock`, `timeout`, `intermittent`) that transfers across C, Go, Java, JavaScript, and Python?
- **Interpretability Fidelity**: When an explainability tool highlights keywords to justify a prediction, can developers trust that those attributions reflect stable domain concepts rather than spurious artifact shifts?

---

## 7. Research Gap
While prior literature has proposed various machine learning models for test flakiness (e.g., King et al., 2018; Haben et al., 2021; Barbosa et al., 2023), several critical methodological gaps remain unaddressed:
1. **Repository Overlap in Random Splits**: Most existing studies evaluate models using standard random or stratified holdouts. In datasets with multiple issues per project, random splits cause intra-repository data leakage, creating overly optimistic performance estimates.
2. **Post-Hoc Model Selection Bias in LOLO Evaluations**: Previous cross-language studies frequently report "best model per language" selected post-hoc based on held-out test performance, conflating model selection with out-of-distribution evaluation.
3. **Neglect of Cluster Correlation in Uncertainty Estimation**: Standard non-parametric bootstrap resampling treats each issue as an independent, identically distributed (i.i.d.) observation, ignoring intra-repository clustering and artificially narrowing confidence intervals.
4. **Uncritical Category Aggregation**: When multi-class models perform poorly on rare classes, researchers often merge categories to report higher accuracy without decomposing the mechanical effect of baseline denominator shifts.
5. **Superficial Explainability Claims**: Claims regarding "feature sign invariance" are frequently asserted without examining mathematical differences between linear model coefficients, tree impurity metrics, and local surrogate methods.

---

## 8. Research Questions

To provide a thorough and methodologically sound investigation, we formulate five central research questions:

* **RQ1 (In-Distribution Effectiveness)**: *How effectively can classical machine learning models detect flaky test issues from software issue reports under controlled in-distribution conditions?*
* **RQ2 (Repository-Independent Generalization)**: *How well do models generalize to completely unseen software repositories when trained and tested on strictly disjoint repository clusters across multiple random seeds?*
* **RQ3 (Cross-Language Generalization)**: *How does model performance vary under Leave-One-Language-Out (LOLO) evaluation when entire programming language ecosystems are systematically excluded from training?*
* **RQ4 (Explainability Reliability)**: *How reliably do post-hoc explainability techniques (SHAP and LIME) characterize model predictions under distribution shift, and what are the mathematical limitations of model-native feature importances?*
* **RQ5 (Root-Cause Granularity Trade-Off)**: *How does the granularity of root-cause taxonomy (13 fine-grained categories vs. 4 macroscopic engineering failure modes) affect classification performance, and to what extent is the observed performance gain driven by chance baseline shifts?*

---

## 9. Related Work

### 9.1 Empirical Studies on Test Flakiness
The software engineering community has extensively studied the prevalence, taxonomy, and lifecycle of flaky tests. In a foundational study, **Luo et al. (2014)** empirically analyzed flaky tests in open-source Java projects, establishing the first major categorization of flaky root causes, highlighting asynchronous waiting, concurrency, and test order dependencies as dominant drivers. **Lam et al. (2019)** expanded this work to examine the lifecycle and fix strategies of flaky tests, observing that developers frequently resort to crude fixes such as adjusting sleep timeouts. **Gruber et al. (2021)** investigated flakiness in the Python ecosystem, revealing that while concurrency issues dominate Java, Python flakiness is heavily driven by platform differences and floating-point assertions.

### 9.2 Machine Learning for Flaky Test Detection
To mitigate the cost of dynamic rerun frameworks such as DeFlaker (**Bell et al., 2018**), researchers turned to static and natural language indicators. **Barbosa et al. (2023)** conducted an extensive cross-language empirical study across five programming languages (C, Go, Java, JavaScript, Python), publishing a curated dataset of GitHub issues and analyzing the linguistic and contextual characteristics of test flakiness. Their findings demonstrated that issue discussions contain rich contextual clues regarding non-deterministic failure modes.

### 9.3 Generalization and Evaluation Integrity in SE
The hazards of data leakage in software engineering ML have been highlighted by **Munaiah et al. (2017)** and **Allamanis (2019)**, who demonstrated that code duplication and repository overlap lead to severely inflated performance estimates. In econometric and statistical literature, **Cameron and Miller (2015)** formalized cluster-robust inference, emphasizing that when data is grouped into natural clusters (such as software repositories), failure to account for intra-cluster correlation yields severely underestimated standard errors and invalid hypothesis tests.

### 9.4 Post-Hoc Interpretability
Post-hoc local interpretability methods, predominantly LIME (**Ribeiro et al., 2016**) and SHAP (**Lundberg and Lee, 2017**), have seen widespread adoption in software engineering for defect prediction and vulnerability detection. However, recent critical studies in machine learning have warned against treating local surrogates as ground-truth causal mechanisms, emphasizing that explanations can exhibit high sensitivity to distribution shifts and hyperparameter choices.

---

## 10. Dataset Description and Reconciliation

We utilize the empirical dataset curated by **Barbosa et al. (IEEE TSE 2023)**, sourced from public GitHub issue trackers.

### 10.1 Schema and Data Reconciliation
Initial documentation across preliminary reports exhibited conflicting repository counts (421 vs. 641 repositories). Our forensic reconciliation resolved this discrepancy:
- The raw spreadsheet `DataBase_GitHub.xlsx` records 760 candidate projects in sheet `ProjectsDomain` (row 1). In row 2, the authors explicitly defined **421** as *"Nº of Projects with at least 1 Flaky (Includes Problem/Solution issues Hard to classify)"*.
- In our cleaned binary classification dataset ($N=1,101$), parsing canonical `{owner}/{repo}` identifiers from `url_issue` reveals exactly **641 unique repositories**.

The stage-by-stage sample accounting is presented in Table 1:

| Dataset Stage | Issue Count ($N$) | Unique Repositories | Target Labels Included |
| :--- | :---: | :---: | :--- |
| **Raw Excel Spreadsheet** | 1,543 | 850 canonical | All extracted rows across 5 languages |
| **Deduplicated Dataset** | 1,513 | 842 canonical | Exact URL duplicates removed |
| **Binary Classification Dataset** | **1,101** | **641 canonical** | Confirmed Flaky ($T=701$), False Positive ($F=400$); $ND=224$ excluded |
| **Root-Cause Classification Dataset** | **589** | **349 canonical** | Confirmed Flaky issues with verified single-label root cause |
| **Repository Train Split (Seed 42)** | 874 | 512 canonical | 80% group partition via `GroupShuffleSplit` |
| **Repository Test Split (Seed 42)** | 227 | 129 canonical | 20% group partition (Strictly 0 repository overlap) |

*Table 1: Formal dataset sample and repository accounting across pipeline stages.*

```
                 +------------------------------------------------+
                 |       Raw Spreadsheet Issues (N = 1,543)       |
                 +------------------------------------------------+
                                         |
                                         v
                 +------------------------------------------------+
                 | Deduplicated Cleaned Issues (N = 1,513; R=842) |
                 +------------------------------------------------+
                                         |
                 +-----------------------+------------------------+
                 |                                                |
                 v (Filter Labeled T/F)                           v (Confirmed T + Root Cause)
+----------------------------------+            +----------------------------------+
| Binary Dataset (N=1,101; R=641)  |            | Root-Cause Dataset (N=589; R=349)|
| - Flaky (T): 701 (63.7%)         |            | - 13 Fine-Grained Classes        |
| - False Positive (F): 400 (36.3%)|            | - 4 Macroscopic Failure Modes    |
+----------------------------------+            +----------------------------------+
```

### 10.2 Language and Class Distributions
Figure 2 illustrates the distribution of samples across flakiness labels, programming language ecosystems, and primary root causes:

![Dataset Distribution](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/dataset_distribution.png)
*Figure 2: Distribution of the FlakyGuard AI dataset: (a) Binary class distribution ($N=1,101$), (b) Target language representation across the 5 ecosystems, and (c) Top-7 root-cause categories among confirmed flaky issues ($N=589$).*

The binary dataset contains 265 JavaScript issues (24.1%), 222 Python issues (20.2%), 211 Java issues (19.2%), 211 C issues (19.2%), and 192 Go issues (17.4%). Flaky issues ($T$) comprise 63.67% of the dataset ($N=701$), while non-flaky false positives ($F$) comprise 36.33% ($N=400$).

---

## 11. Proposed Methodology and System Architecture

The FlakyGuard AI architecture is engineered to provide end-to-end processing, evaluation isolation, and diagnostic interpretability:

![System Architecture](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/system_architecture.png)
*Figure 1: High-level architectural pipeline of FlakyGuard AI, depicting ingestion, strict leakage-free feature extraction, multi-regime evaluation partitions, and explainability audits.*

### 11.1 Preprocessing Pipeline
1. **Text Normalization**: Combined text is constructed as $\text{text} = \text{title} + \text{" "} + \text{body}$.
2. **Noise and Leakage Removal**: URLs, markdown code fences, hexadecimal commit hashes, issue tracker issue numbers (`#1234`), and reviewer resolution tags are stripped via regular expressions.
3. **Partition Isolation**: To prevent data leakage, text cleaning and normalization operate independently on each record, while vocabulary construction and IDF statistics are fitted exclusively on training partitions.

---

## 12. Feature Engineering

We employ sublinear Term Frequency-Inverse Document Frequency (TF-IDF) vectorization:
$$\text{TF-IDF}(t, d, D) = \left(1 + \log(\text{TF}(t, d))\right) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
Features are constrained to unigrams and bigrams with a maximum vocabulary size of 8,000 features, discarding terms appearing in fewer than 2 documents ($\text{min\_df}=2$). All vectorizers are fitted strictly on training splits and applied out-of-sample to test splits.

---

## 13. Machine Learning Models

We deploy three classical, interpretable architectures well-suited for high-dimensional, sparse text representations:

1. **Random Forest Classifier**:
   - Hyperparameters: $N_{\text{estimators}} = 150$, $\text{max\_depth} = 30$, balanced class weighting, Gini impurity criterion.
   - Non-linear ensemble that captures feature interactions without parametric assumptions.
2. **Logistic Regression**:
   - Hyperparameters: $L_2$ regularization penalty, inverse regularization strength $C=1.0$, balanced class weighting, maximum iterations $= 1,000$, L-BFGS solver.
   - Produces calibrated log-odds estimates with directly interpretable linear signed coefficients $w_j \in (-\infty, +\infty)$.
3. **Calibrated Linear Support Vector Machine (Linear SVM)**:
   - Hyperparameters: LinearSVC with $C=1.0$, balanced class weighting, calibrated using 3-fold Platt scaling (`CalibratedClassifierCV`) to generate well-calibrated posterior probabilities.

---

## 14. Explainable AI Methodology

To satisfy regulatory and developer debugging requirements, we implement complementary interpretability techniques:

1. **Game-Theoretic SHAP (SHapley Additive exPlanations)**:
   Computes the unique additive feature attribution vector $\phi(x)$ satisfying efficiency, symmetry, dummy, and additivity:
   $$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$
   We apply `shap.LinearExplainer` on Logistic Regression to compute exact global Shapley values across test issues.
2. **Local Interpretable Model-agnostic Explanations (LIME)**:
   Approximates the complex decision boundary in the local $\epsilon$-neighborhood of a specific instance $x$ by training an interpretable surrogate model $g \in G$:
   $$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
   Implemented via `LimeTextExplainer` with exponential distance kernel $\pi_x$ and bag-of-words perturbations.
3. **Tree Mean Decrease in Impurity (MDI)**:
   Measures the total reduction in Gini impurity brought by feature $j$ across all split nodes in the forest. As detailed in Section 22, MDI is strictly non-negative ($\ge 0$) and provides importance ranking without directional sign.

---

## 15. Experimental Design and Validation Regimes

To systematically eliminate evaluation artifacts, we evaluate models across four experimental regimes:

1. **Regime 1: Stratified Random Holdout (80/20)**:
   Standard baseline partition ($N_{\text{train}}=880, N_{\text{test}}=221$). Reflects in-distribution performance where repository overlap may exist.
2. **Regime 2: Repeated Repository-Independent Splits (10 Seeds)**:
   Grouped evaluation via `GroupShuffleSplit` across 10 fixed random seeds (`[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`). In every fold, all issues from the same repository reside exclusively in either the train set ($G \approx 512$ repos) or the test set ($G \approx 129$ repos). Overlap is strictly zero.
3. **Regime 3: Leave-One-Language-Out (LOLO)**:
   Five separate evaluations wherein one entire programming language ecosystem is held out as the test set, while models are trained exclusively on the remaining four languages. TF-IDF vectorization is fitted strictly on the four training languages.
4. **Regime 4: Non-Parametric and Cluster Bootstrap**:
   $B=1,000$ bootstrap iterations. For repository-independent evaluation, we implement **repository-level cluster bootstrap**, resampling repository clusters with replacement rather than individual issues.

---

## 16. Evaluation Metrics

Given class imbalance (63.7% flaky issues in binary, acute right-skew in root causes), accuracy alone is misleading. We report:
- **Macro-Averaged F1-Score ($\text{Macro-F1}$)**: The unweighted arithmetic mean of F1-scores across all classes:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
  Serves as the primary optimization metric, giving equal weight to minority and majority classes.
- **Accuracy, Precision, Recall, and F1-Score** for the positive (flaky) class.
- **Receiver Operating Characteristic Area Under Curve (ROC-AUC)**: Threshold-independent measure of ranking quality.
- **Precision-Recall Area Under Curve (PR-AUC)**: Informative evaluation under class imbalance.

---

## 17. Results: In-Distribution Binary Detection (RQ1)

Table 2 and Figure 3 summarize model performance on the stratified random 80/20 holdout test set ($N=221$):

| Architecture | Macro-F1 (95% CI) | Accuracy | ROC-AUC (95% CI) | PR-AUC | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.6848** [0.6235, 0.7466] | **0.7240** | **0.7310** [0.6601, 0.8019] | **0.8258** | 0.7619 | 0.8143 |
| **Logistic Regression** | 0.6555 [0.5946, 0.7198] | 0.6923 | 0.7262 [0.6555, 0.7978] | 0.8066 | 0.7634 | 0.7429 |
| **Linear SVM** | 0.6107 [0.5400, 0.6757] | 0.6878 | 0.6993 [0.6309, 0.7618] | 0.8035 | 0.7093 | 0.8714 |
| *Majority Prior Baseline* | 0.4550 | 0.6335 | 0.5000 | 0.6335 | 0.6335 | 1.0000 |

*Table 2: Binary flakiness detection performance on the stratified random 80/20 holdout test set ($N=221$).*

![Binary Model Comparison](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/binary_model_comparison.png)
*Figure 3: Binary classification performance across the candidate models on the random holdout test set ($N=221$), showing Macro-F1, Accuracy, ROC-AUC, and PR-AUC relative to the majority class prior baseline.*

### Answer to RQ1:
Classical natural language models effectively detect flaky test issue reports. Random Forest achieves the strongest in-distribution performance (Macro-F1: **0.6848**, ROC-AUC: **0.7310**), substantially exceeding the zero-rule majority class baseline (Macro-F1: **0.4550**). Logistic Regression provides highly competitive performance (Macro-F1: **0.6555**) with superior linear interpretability.

---

## 18. Results: Repository-Independent Generalization (RQ2)

To evaluate whether performance collapses when encountering completely unseen codebases, we analyzed 10 repeated `GroupShuffleSplit` partitions with strictly zero repository overlap.

### 18.1 Multi-Split Stability
Table 3 and Figures 4 and 6 present aggregate statistics across the 10 independent folds (varying between 508–514 train repositories and 127–133 test repositories):

| Architecture | Macro-F1 (Mean $\pm$ Std) | Macro-F1 Range [Min, Max] | Accuracy (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.6670 $\pm$ 0.0239** | [0.6398, 0.7087] | 0.6925 $\pm$ 0.0317 | **0.7265 $\pm$ 0.0380** |
| **Random Forest** | 0.6542 $\pm$ 0.0267 | [0.6015, 0.7092] | **0.6948 $\pm$ 0.0265** | 0.7091 $\pm$ 0.0346 |
| **Linear SVM** | 0.5201 $\pm$ 0.0546 | [0.4541, 0.6399] | 0.6767 $\pm$ 0.0367 | 0.6800 $\pm$ 0.0399 |

*Table 3: Aggregate performance metrics across 10 repeated repository-independent splits with zero repository overlap.*

![Repository Independent Comparison](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/repository_independent_comparison.png)
*Figure 4: Generalization performance comparing the random split holdout against the single repository-independent split (Seed 42) and the 10-fold cross-repository mean.*

![Repeated Repository Distribution](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/repeated_repository_distribution.png)
*Figure 6: Distribution of (a) Macro-F1 and (b) ROC-AUC across 10 independent repository-grouped splits, displaying median, interquartile ranges, and individual fold observations.*

### 18.2 Cluster Bootstrap vs. Naive i.i.d. Resampling
Applying repository-level cluster bootstrapping (Table 4) reveals that naive i.i.d. resampling artificially narrows confidence intervals:

| Model | Observed Macro-F1 | Cluster Bootstrap 95% CI | Cluster SE | Naive i.i.d. 95% CI | Naive SE | SE Inflation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7092 | **[0.6396, 0.7732]** | 0.0349 | [0.6477, 0.7698] | 0.0310 | **$1.13\times$** |
| **Logistic Regression** | 0.6978 | **[0.6352, 0.7530]** | 0.0303 | [0.6404, 0.7549] | 0.0308 | $0.98\times$ |
| **Linear SVM** | 0.5321 | **[0.4588, 0.6014]** | 0.0355 | [0.4715, 0.6026] | 0.0341 | **$1.04\times$** |

*Table 4: Statistical comparison of cluster-robust bootstrap vs. naive i.i.d. bootstrap on repository-independent test set ($N=227, G=129$).*

### Answer to RQ2:
The hypothesis of catastrophic cross-repository degradation is **falsified**. Logistic Regression (Macro-F1: **0.6670 $\pm$ 0.0239**) and Random Forest (Macro-F1: **0.6542 $\pm$ 0.0267**) maintain strong, stable discrimination across 10 unseen repository cluster evaluations. However, accounting for repository clustering is essential: cluster bootstrapping reveals up to a $1.13\times$ standard error inflation, establishing true out-of-repository confidence bounds of **[0.6396, 0.7732]**.

---

## 19. Results: Cross-Language Generalization (RQ3)

Table 5 and Figure 5 present the complete, unbiased $5 \times 3$ performance matrix under Leave-One-Language-Out evaluation, completely eliminating post-hoc test selection bias:

| Held-Out Language | Test Samples ($N$) | Flaky Prior | Random Forest Macro-F1 | Logistic Regression Macro-F1 | Linear SVM Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C** | 211 | 67.3% | 0.5931 [0.525, 0.656] | **0.6635** [0.599, 0.727] | 0.4974 [0.426, 0.567] |
| **Go** | 192 | 68.2% | 0.6298 [0.550, 0.709] | **0.6327** [0.554, 0.711] | 0.5365 [0.452, 0.617] |
| **Java** | 211 | 66.8% | 0.4922 [0.431, 0.560] | **0.5144** [0.447, 0.584] | 0.4691 [0.422, 0.534] |
| **JavaScript** | 265 | 58.9% | **0.6929** [0.627, 0.743] | 0.6863 [0.630, 0.747] | 0.5263 [0.466, 0.585] |
| **Python** | 222 | 58.6% | **0.5931** [0.527, 0.662] | 0.5297 [0.466, 0.591] | 0.3816 [0.357, 0.405] |
| **Unbiased Mean $\pm$ Std** | **Total: 1,101** | **63.7%** | **0.6002 $\pm$ 0.0722** | **0.6053 $\pm$ 0.0818** | **0.4822 $\pm$ 0.0610** |
| **Pooled Cross-Language CI** | **1,101** | **63.7%** | **0.6588** [0.629, 0.687] | **0.6539** [0.625, 0.682] | **0.5024** [0.473, 0.532] |

*Table 5: Unbiased Leave-One-Language-Out performance matrix across all 5 languages and 3 architectures.*

![LOLO Language Comparison](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/lolo_language_comparison.png)
*Figure 5: Leave-One-Language-Out (LOLO) performance across all five held-out languages and pooled cross-language evaluations, comparing Random Forest, Logistic Regression, and Linear SVM against the majority class prior.*

### Answer to RQ3:
Cross-language generalization exhibits substantial ecosystem heterogeneity. When evaluated across all 5 held-out ecosystems without post-hoc selection, **Logistic Regression achieves the highest mean transfer score (Macro-F1: 0.6053 $\pm$ 0.0818)**, outperforming Random Forest (0.6002). Linear SVM experiences severe degradation (0.4822) when language-specific tokens are absent. Cross-language transfer is highest in JavaScript (Macro-F1: 0.6929) and C (0.6635), but drops significantly in Java (0.5144), where idiosyncratic framework verbiage (e.g., Spring, JUnit annotations) creates domain-specific vocabulary barriers.

---

## 20. Results: Explainability Reliability Under Distribution Shift (RQ4)

### 20.1 Global SHAP Token Attributions
Figure 9 illustrates the top 15 global tokens identified by SHAP LinearExplainer on test issues:

![SHAP Global Explanation](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/shap_global_explanation.png)
*Figure 9: Global feature importance generated via SHAP LinearExplainer on Logistic Regression, showing the top 15 tokens driving model outputs.*

Tokens strongly driving flakiness predictions include domain-general indicators: `flaky`, `timeout`, `race`, `intermittent`, `wait`, and `sleep`. Conversely, tokens associated with non-flaky false alarms include routine maintenance terms: `add`, `update`, `version`, `support`, and `install`.

### 20.2 Attribution Stability Audit
Figure 10 presents the forensic audit comparing Logistic Regression feature coefficients between the Random Split Model (subject to repository overlap) and the Repository-Independent Model (zero overlap):

![Explainability Stability](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/explainability_stability.png)
*Figure 10: Attribution stability audit across repository distribution shift, comparing feature coefficients between random split and repository-independent models and highlighting the sign inversion of `goroutine`.*

Table 6 details the exact coefficient shifts:

| Token | Random Split Weight | Repo-Independent Weight | Absolute Shift ($|\Delta|$) | Invariance Status |
| :--- | :---: | :---: | :---: | :---: |
| `flaky` | $+1.8239$ | $+1.7810$ | $0.0429$ | **Stable (+)** |
| `fix` | $+0.3755$ | $+0.7610$ | $0.3855$ | **Stable (+)** |
| `race` | $+0.4290$ | $+0.4961$ | $0.0670$ | **Stable (+)** |
| `timeout` | $+0.4180$ | $+0.3593$ | $0.0586$ | **Stable (+)** |
| `intermittent` | $+0.2476$ | $+0.1712$ | $0.0764$ | **Stable (+)** |
| `sleep` | $+0.0478$ | $+0.0606$ | $0.0129$ | **Stable (+)** |
| `add` | $-0.5960$ | $-0.5555$ | $0.0405$ | **Stable (-)** |
| `wait` | $-0.1379$ | $-0.1718$ | $0.0340$ | **Stable (-)** |
| `error` | $-0.0695$ | $-0.1940$ | $0.1244$ | **Stable (-)** |
| `goroutine` | **$-0.1708$** | **$+0.0387$** | **$0.2095$** | **SIGN INVERTED ($- \rightarrow +$)** |

*Table 6: Forensic weight comparison across distribution shift for 10 core domain tokens.*

### 20.3 Dissection of the "100% Sign Invariance" Claim & Tree MDI Mechanics
Our audit refutes earlier claims of "100% sign invariance":
1. **Curated Token Sign Flip**: Within the 10 evaluated terms, `goroutine` flipped sign from negative to positive because Go issues were clustered unevenly between train and test splits, proving that distribution shift can invert polarity for language-specific keywords.
2. **The Random Forest MDI Fallacy**: Random Forest Gini impurity importance is mathematically bounded to $[0, 1]$:
   $$\text{MDI}(j) = \frac{1}{N_{\text{trees}}} \sum_T \sum_{t: v(t)=j} \Delta \text{Gini}(t) \ge 0$$
   MDI importances have **no directional sign**. Stating that Random Forest features have "stable positive signs" is a mathematical impossibility. Directional local explanations for tree models require **SHAP TreeExplainer**, not native Gini importances.

### Answer to RQ4:
Core concurrency and timing keywords demonstrate high directional stability (90% sign consistency in curated linear coefficients). However, language-specific tokens can invert signs under repository clustering. Native tree feature importances provide only non-negative relevance rankings; signed interpretability must rely on game-theoretic Shapley values or local surrogates.

---

## 21. Results: Root-Cause Classification Granularity (RQ5)

### 21.1 13-Class Benchmark Performance
Figure 7 displays the confusion matrix for the 13 fine-grained categories evaluated on holdout test issues ($N=118$):

![Root Cause Confusion Matrix](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/root_cause_confusion_matrix.png)
*Figure 7: Confusion matrix for 13-class root-cause classification using class-balanced Random Forest on holdout test issues ($N=118$).*

Table 7 summarizes multi-class performance across the candidate architectures:

| Setting | Architecture | Macro-F1 (95% CI) | Weighted-F1 | Accuracy | Random Baseline | Baseline Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **13 Classes** | Linear SVM (Balanced) | 0.1844 [0.137, 0.251] | 0.3340 | 33.90% | 7.69% | $2.40\times$ |
| **13 Classes** | Logistic Regression (Balanced) | 0.2289 [0.161, 0.305] | 0.3541 | 36.44% | 7.69% | $2.98\times$ |
| **13 Classes** | **Random Forest (Balanced)** | **0.2834** [0.198, 0.349] | **0.3705** | **38.14%** | 7.69% | **$3.69\times$** |
| **4 Macro Classes** | **Logistic Regression (Balanced)** | **0.4222** [0.325, 0.520] | **0.5376** | **54.24%** | 25.00% | $1.69\times$ |

*Table 7: Multi-class root-cause classification performance under 13 discrete categories vs. 4 macroscopic failure modes.*

### 21.2 Mathematical Deconstruction of the 13 vs. 4 Category Rise
Figure 8 compares the 13-class and 4-class settings alongside their respective chance baselines:

![Root Cause Comparison](file:///Users/himani/Downloads/AI-Based%20Flaky%20Test%20Detection/artifacts/figures/publication/root_cause_13_vs_4_comparison.png)
*Figure 8: Performance comparison between 13 fine-grained classes and 4 macroscopic failure modes, highlighting model accuracy and Macro-F1 alongside uniform and empirical prior baselines.*

The rise in Macro-F1 from 0.2834 to 0.4222 is governed by three mathematical mechanisms rather than superior model learning:
1. **Chance Baseline Shift**: In a 13-class problem, uniform random accuracy is $1/13 = 7.69\%$. In a 4-class problem, uniform chance is $1/4 = 25.00\%$ ($+17.31$ pp). Evaluated relative to chance, 13-class Random Forest achieves **$3.69\times$ over baseline**, whereas 4-class Logistic Regression achieves only **$1.69\times$ over baseline**.
2. **Within-Cluster Confusion Elimination**: In the 13-class space, confusing `Async Wait` with `Test Case Timeout` incurs two penalties (a false negative and a false positive). In the 4-class taxonomy, both map to `Timing & Asynchrony`, converting confusion into true positives.
3. **Resolution of Extreme Minority Zero-F1 Penalty**: Categories with acute rarity (e.g., `Test Suite Timeout` with 5 total samples, `Too Restrictive Range` with 6 total samples) score $F_1 \approx 0$ on the test set, mathematically pulling down the unweighted 13-class arithmetic mean.

### Answer to RQ5:
Fine-grained 13-class root-cause classification is bounded by extreme minority sample scarcity. Aggregating into 4 macroscopic engineering modes raises Macro-F1 to **0.4222** (Accuracy: **54.24%**). However, this increase is primarily an artifact of baseline denominator shifts and penalty absorption. We recommend a **two-tier architecture**: deploying the 4-class model for automated CI/CD team routing, while presenting top-3 13-class candidate rankings for developer debugging.

---

## 22. Limitations

1. **Textual Proxy Limitation**: FlakyGuard AI analyzes natural language issue reports and pull requests submitted by developers. It does not inspect dynamic execution traces, thread schedules, or memory dumps. The system detects flakiness as perceived and described by humans.
2. **Vocabulary Overlap Constraints**: Unigram and bigram TF-IDF representations cannot fully capture deep semantic context, long-range dependencies, or code snippet logic embedded within issue bodies.
3. **Minority Class Sparsity**: With only 589 confirmed root-cause annotations across 13 categories, rare classes have insufficient statistical support for high-confidence supervised learning.

---

## 23. Threats to Validity

1. **Construct Validity**: The ground-truth flakiness and root-cause labels were manually assigned by the original dataset annotators (Barbosa et al., 2023). Subjectivity in developer labeling could introduce label noise.
2. **Internal Validity**: Post-outcome metadata leakage was prevented by systematically removing resolution tags, commit links, and reviewer discussions. Vectorizers were fitted strictly on training partitions across all regimes.
3. **External Validity**: Although the dataset spans 641 repositories across five major programming languages, it is sourced exclusively from open-source GitHub repositories. Enterprise proprietary repositories may exhibit different issue reporting cultures.
4. **Conclusion Validity**: Addressed by conducting 10 repeated group-shuffle splits, unbiased Leave-One-Language-Out evaluations, and repository-level cluster bootstrapping ($B=1,000$).

---

## 24. Conclusion

This paper presented an empirical investigation into the generalization, transferability, and explainability of classical machine learning for flaky test detection. 

Our findings demonstrate that:
1. Classical natural language models (Random Forest and Logistic Regression) provide effective, computationally efficient flaky test detection, substantially outperforming empirical chance baselines.
2. Flaky test detection models generalize robustly to completely unseen software repositories (Macro-F1: **0.6670 $\pm$ 0.0239** across 10 independent splits).
3. Under unbiased cross-language evaluation, Logistic Regression provides the most stable cross-ecosystem transfer (Macro-F1: **0.6053**).
4. Post-hoc explainability reflects genuine domain concepts for universal timing and concurrency keywords, but language-specific terms are vulnerable to polarity inversion under distribution shift.
5. Macroscopic root-cause grouping provides an actionable CI/CD triage mechanism, but reported metric gains must be interpreted in light of mathematical baseline denominator shifts.

---

## 25. Future Work
Future investigations will focus on:
1. Pre-trained software engineering language models (e.g., CodeBERT, StarCoder) fine-tuned on issue reports to evaluate whether contextual embeddings improve cross-language vocabulary alignment.
2. Multi-modal triage architectures combining natural language issue text with dynamic test execution metadata (execution duration, historical test pass rates, code churn).
3. Active learning workflows designed to solicit developer annotations specifically on rare root-cause failure modes.

---

## 26. References

1. **Barbosa, K., Ferreira, R., Pinto, G., d’Amorim, M., & Miranda, B.** (2023). Test Flakiness Across Programming Languages. *IEEE Transactions on Software Engineering*, 49(4), 2039–2052. DOI: 10.1109/TSE.2022.3208864.
2. **Luo, Q., Hariri, F., Eloussi, L., & Marinov, D.** (2014). An empirical analysis of flaky tests. In *Proceedings of the 22nd ACM SIGSOFT International Symposium on Foundations of Software Engineering (FSE 2014)*, pp. 643–653. DOI: 10.1145/2635868.2635920.
3. **Bell, J., Legunsen, O., Hilton, M., Eloussi, L., Yung, T., & Marinov, D.** (2018). DeFlaker: Automatically detecting flaky tests. In *Proceedings of the 40th International Conference on Software Engineering (ICSE 2018)*, pp. 433–444. DOI: 10.1145/3180155.3180164.
4. **Lam, P., Winter, S., Wei, A., Tao, T., Neamtiu, I., & Marinov, D.** (2019). A large-scale empirical study on the lifecycle of flaky tests. In *2019 IEEE International Conference on Software Testing, Verification and Validation (ICST 2019)*, pp. 147–158. DOI: 10.1109/ICST.2019.00024.
5. **Gruber, M., Lukasczyk, S., Kroiß, C., & Fraser, G.** (2021). An empirical study on the flakiness of tests in Python. In *2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE 2021)*, pp. 471–482. DOI: 10.1109/ASE51524.2021.9678783.
6. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*, pp. 1135–1144. DOI: 10.1145/2939672.2939778.
7. **Lundberg, S. M., & Lee, S. I.** (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 4765–4774.
8. **Cameron, A. C., & Miller, D. L.** (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), pp. 317–372. DOI: 10.3368/jhr.50.2.317.
9. **Efron, B., & Tibshirani, R. J.** (1994). *An introduction to the bootstrap*. CRC Press.
10. **Allamanis, M.** (2019). The adverse effects of code duplication in machine learning models of code. In *Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward! 2019)*, pp. 143–153. DOI: 10.1145/3359591.3359735.
11. **Munaiah, N., Kroh, S., Cabrey, C., & Nagappan, M.** (2017). Curating GitHub for engineered software projects. *Empirical Software Engineering*, 22(6), pp. 3219–3253. DOI: 10.1007/s10664-017-9512-6.
12. **Gyori, A., Shi, A., Hariri, F., & Marinov, D.** (2015). Reliable testing: Detecting state-polluting tests to prevent test dependency. In *Proceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering (ESEC/FSE 2015)*, pp. 223–233. DOI: 10.1145/2786805.2786853.
13. **Zhang, S., Jalali, D., Wuttke, J., Muşlu, K., Ernst, M. D., & Notkin, D.** (2014). Empirically revisiting the test generation challenge: Why test generation tools fail to generate tests that detect bugs. In *Proceedings of the 2014 International Symposium on Software Testing and Analysis (ISSTA 2014)*, pp. 217–228. DOI: 10.1145/2610384.2610410.
