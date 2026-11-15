# FlakyGuard AI: Faculty Supervisor & Academic Defense Q&A

This document prepares the researcher for rigorous technical examination by faculty supervisors, thesis committees, and academic peer reviewers. It contains **44 challenging questions** paired with mathematically grounded, evidence-backed answers.

---

## Part 1: Problem Motivation & Domain Relevance

### Q1: Why do flaky tests matter in industrial software engineering?
**Answer**: Flaky tests pass and fail non-deterministically without code changes. In CI/CD pipelines, they destroy developer trust in automated test suites, waste expensive compute hours on re-runs, block release trains, and cause developers to ignore legitimate bug failures, introducing severe regression hazards into production.

### Q2: How does your work differ from existing tools like DeFlaker?
**Answer**: Tools like DeFlaker (Bell et al., ICSE 2018) require dynamic execution traces, code instrumentation, and repeated test execution inside the CI runtime. FlakyGuard AI addresses a complementary problem: triaging natural language issue reports and pull requests submitted by developers. It operates post-hoc at the issue-tracker layer, enabling rapid triage and root-cause classification before an engineer begins reproducing the failure.

### Q3: Why analyze GitHub issues rather than runtime CI execution logs?
**Answer**: Runtime CI logs are voluminous, proprietary, retention-limited, and often uncurated. Conversely, GitHub issues represent developer-validated ground truth where engineers have confirmed that an observed test failure was truly flaky or a false positive, accompanied by root-cause explanations and fix discussions.

### Q4: If the system analyzes developer text, doesn't it require a human to write an issue first?
**Answer**: Yes. FlakyGuard AI is explicitly scoped as an **issue-triage and diagnostic assistant**, not an in-process dynamic testing hook. It automates label verification, false-positive filtering, and root-cause categorization when issues are filed.

---

## Part 2: Dataset Construction, Provenance & Cleaning

### Q5: What is the provenance of your dataset?
**Answer**: The dataset comes from the peer-reviewed benchmark by Barbosa et al. (*IEEE TSE 2023*, "Test Flakiness Across Programming Languages", benchmark repo: `Test-Flaky/TSE22`). It was collected by querying closed GitHub issues labeled `bug` containing `"flaky"` and `"test"` across five languages.

### Q6: How many raw samples are in the dataset, and how did you clean them?
**Answer**: The raw spreadsheet contains 1,543 records. We removed 30 exact URL and text duplicates, resulting in 1,513 deduplicated issues. We filtered out 410 "Not Decided" (`ND`) issues, yielding **1,101 binary classification samples** (701 Flaky, 400 False Positive). For root-cause classification, we filtered to the **589 confirmed flaky issues** annotated across 13 classes.

### Q7: Why did earlier reports cite 421 repositories, but your dataset contains 641?
**Answer**: The figure 421 originates from row 2 of the author's metadata tab `ProjectsDomain` in `DataBase_GitHub.xlsx` ("Nº of Projects with at least 1 Flaky"). That count reflects the authors' specific domain-filtering criteria. In contrast, parsing all valid issue URLs in the 1,101 binary dataset via regex (`github.com/{owner}/{repo}`) reveals **641 unique canonical repositories**.

### Q8: How did you handle 192 rows with missing repository names?
**Answer**: While 192 rows had an empty `Repository/Project` cell, every single row possessed an intact `URL Issue` link. We implemented `parse_canonical_repo`, extracting `{owner}/{repo}` directly from the URL.

### Q9: Did you ensure there is no target leakage in your preprocessing?
**Answer**: Yes. Columns containing reviewer annotations (`Reviewer 1`, `Reviewer 2`), post-hoc categorization, fix strategies, and solution notes were strictly pruned. Features are extracted solely from `title`, `body`, and `language`.

---

## Part 3: Feature Representation & Why TF-IDF Over Deep Learning

### Q10: Why use classical TF-IDF rather than pre-trained transformers like CodeBERT or RoBERTa?
**Answer**: Three scientific and practical reasons:
1. **Interpretability**: Linear models over TF-IDF provide exact, monotonic mathematical attributions ($f(x) = \mathbf{w}^T \mathbf{x} + b$), whereas transformer attention weights are non-monotonic and do not reliably reflect input feature importance.
2. **Computational Reproducibility**: TF-IDF trains in $<10$ seconds on a standard laptop CPU without GPU hardware, enabling fast repeated partitions ($10 \times$) and bootstrap resamples ($B=1,000$).
3. **Dataset Scale**: With 1,101 samples, deep language models with hundreds of millions of parameters risk severe representation overfitting unless extensively regularized.

### Q11: What specific TF-IDF configuration did you use?
**Answer**: We extracted unigrams and bigrams (`ngram_range=(1, 2)`), applied sublinear term-frequency scaling ($1 + \log(\text{tf})$), excluded tokens with document frequency $<2$ or $>0.85$, and bounded the vocabulary to 5,000 features for binary detection and 3,000 for root cause.

### Q12: How do you prevent vocabulary leakage during vectorization?
**Answer**: In all generalization experiments (Repository-Grouped and LOLO), the `TfidfVectorizer` is fitted **exclusively on the training partition**. Test partition documents are transformed using the fitted training vectorizer. Unseen test tokens receive zero weights without error.

### Q13: Does TF-IDF understand code syntax in embedded code snippets?
**Answer**: No. TF-IDF treats code snippets as lexical bags of tokens. While technical tokens like `assert`, `timeout`, and `goroutine` are captured, Abstract Syntax Tree (AST) grammar and execution paths are not modeled. This is documented as a known limitation.

### Q14: Why did you prepend the programming language to the text?
**Answer**: Language context disambiguates overloaded tokens. For example, `wait` in Go often relates to `sync.WaitGroup`, while in JavaScript it relates to `async/await` promise resolution.

---

## Part 4: Machine Learning Algorithms & Model Selection

### Q15: Why evaluate Random Forest, Logistic Regression, and Linear SVM?
**Answer**: They represent three distinct mathematical inductive biases:
- **Logistic Regression**: Linear decision boundary with probabilistic log-odds calibration and L2 regularization.
- **Linear SVM**: Maximum-margin hyperplane separation optimized for high-dimensional sparse text.
- **Random Forest**: Non-linear ensemble of orthogonal decision trees capturing high-order feature interactions.

### Q16: Why did Random Forest perform best in-distribution, but Logistic Regression performed best out-of-distribution?
**Answer**: Random Forest exploits high-order non-linear token interactions that are effective when training and test sets share vocabulary (in-distribution). However, under repository or language distribution shift, unobserved token combinations cause tree splits to fail abruptly. Logistic Regression distributes weights globally and smoothly across all active features, yielding superior robustness under distribution shift ($0.6670$ vs $0.6542$).

### Q17: Why did Linear SVM collapse under repository and language shift?
**Answer**: Linear SVM constructs an uncalibrated maximum-margin hyperplane that relies heavily on support vectors near the boundary. When repository shift removes project-specific boundary tokens, the margin collapses, dropping Macro-F1 to $0.5201$ (repository) and $0.4822$ (cross-language).

### Q18: How were hyperparameters chosen, and was there data leakage in tuning?
**Answer**: Hyperparameters (L2 penalty $C=1.0$, 100 trees, Platt scaling calibration) were selected using cross-validation restricted strictly to training folds.

---

## Part 5: Evaluation Metrics & Macro-F1 vs Accuracy

### Q19: Why evaluate using Macro-F1 rather than Accuracy?
**Answer**: The dataset exhibits class imbalance (63.7% Flaky vs 36.3% False Positive in Task A; 17.8% `Async Wait` vs 0.85% `Test Suite Timeout` in Task B). A naive majority-class classifier achieves 63.35% accuracy in Task A and 37.29% accuracy in 4-class Task B while having near-zero diagnostic value. Macro-F1 computes the unweighted arithmetic mean of per-class F1-scores:
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F_1^{(k)}$$
giving equal weight to all classes and penalizing models that ignore minority classes.

### Q20: What was the error in earlier baseline comparisons, and how did you fix it?
**Answer**: Earlier drafts divided Macro-F1 by the uniform accuracy baseline ($1/K$), falsely claiming a "$3.69\times$ lift" ($0.2834 / 0.0769$). This was mathematically invalid because Macro-F1 and Accuracy are distinct metrics. We implemented `src/root_cause_baselines.py` to simulate 1,000 Monte Carlo trials, computing empirical Macro-F1 and Accuracy baselines separately.

### Q21: What are the true empirical baselines for 13-class root-cause classification?
**Answer**: On the test set ($N=118$):
- **Uniform-Random Baseline**: Macro-F1 = **0.0655**, Accuracy = **7.76%**
- **Majority-Class Baseline (`Async Wait`)**: Macro-F1 = **0.0232**, Accuracy = **17.80%**
- **Stratified Prior Baseline**: Macro-F1 = **0.0741**, Accuracy = **11.72%**
- **Model Performance (Random Forest)**: Macro-F1 = **0.2834** ($4.33\times$ lift), Accuracy = **37.29%** ($4.81\times$ lift).

### Q22: What are the true empirical baselines for 4-class macro root-cause classification?
**Answer**:
- **Uniform-Random Baseline**: Macro-F1 = **0.2383**, Accuracy = **25.07%**
- **Majority-Class Baseline (`Resource & Env`)**: Macro-F1 = **0.1358**, Accuracy = **37.29%**
- **Stratified Prior Baseline**: Macro-F1 = **0.2446**, Accuracy = **28.39%**
- **Model Performance (Logistic Regression)**: Macro-F1 = **0.4222** ($1.77\times$ lift), Accuracy = **48.31%** ($1.93\times$ lift).

---

## Part 6: Repository-Independent Generalization & GroupShuffleSplit

### Q23: Why is repository-level splitting necessary?
**Answer**: Standard IID random splitting allows issues from the same repository to appear in both training and test sets (35.3% repository overlap in standard splits). Models can memorize project-specific identifiers (e.g. `couchbase`, `walrus`, project maintainer handles) rather than learning genuine flakiness indicators, producing overly optimistic performance estimates.

### Q24: Why are your 10 evaluations called "repeated random partitions" rather than "10-fold cross validation"?
**Answer**: `GroupShuffleSplit` randomly allocates 80% of repositories to training and 20% to testing on each repetition. While zero repository overlap is strictly enforced *within each run*, test repositories overlap across runs. Calling them "independent folds" or "cross-validation" is methodologically inaccurate; they are **10 repeated repository-grouped random partitions**.

### Q25: Did any repository leak between train and test across your 10 seeds?
**Answer**: No. Automated test `test_zero_repository_overlap_across_all_10_seeds` verifies that for all 10 seeds (`[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`):
$$\text{Train Repos} \cap \text{Test Repos} = \emptyset$$

### Q26: How much does performance drop when moving from in-distribution to repository-independent evaluation?
**Answer**: The drop is modest. In-distribution Random Forest achieves Macro-F1 = $0.6848$, while across 10 repository-grouped partitions Random Forest achieves Mean Macro-F1 = $0.6542 \pm 0.0267$ and Logistic Regression achieves $0.6670 \pm 0.0239$. This demonstrates that core flakiness vocabulary generalizes across repositories.

### Q27: How many repositories were in the training and test sets?
**Answer**: For each seed, approximately $512 \pm 3$ repositories were assigned to training and $129 \pm 3$ repositories to testing, strictly totaling 641 canonical repositories.

---

## Part 7: Cross-Language Transfer & Leave-One-Language-Out

### Q28: How does the Leave-One-Language-Out (LOLO) experiment work?
**Answer**: For each of the five languages (C, Go, Java, JavaScript, Python), all issues belonging to that language are held out as the test set. Models and vectorizers are trained strictly on the remaining four languages.

### Q29: What is the difference between language-mean Macro-F1 and pooled LOLO Macro-F1?
**Answer**:
- **Language-Mean Macro-F1**: The unweighted average of the five individual language test set Macro-F1 scores ($0.6053 \pm 0.0785$ for LR; $0.6002 \pm 0.0729$ for RF; $0.4822 \pm 0.0621$ for SVM).
- **Pooled LOLO Macro-F1**: Macro-F1 computed globally across all 1,101 test predictions concatenated together ($0.6588$ for RF; $0.6539$ for LR; $0.5024$ for SVM).

### Q30: Why is JavaScript transfer highest and Java transfer lowest?
**Answer**: JavaScript transfer is highest (Macro-F1 $\approx 0.693$) because asynchronous timing keywords (`async`, `await`, `timeout`, `promise`) are prominent across all language test suites. Java transfer is lowest (Macro-F1 $\approx 0.515$). However, we treat this as an **unisolated observational hypothesis** because language boundaries correlate with project domains, issue templates, and assertion frameworks.

### Q31: Did vocabulary from the held-out language leak into training?
**Answer**: No. TF-IDF vectorization was fitted exclusively on the 4 training languages. Any syntax or keywords exclusive to the held-out language were unobserved during training.

---

## Part 8: Statistical Bootstrap, Clustering & Uncertainty

### Q32: Why use cluster bootstrap instead of naive IID bootstrap?
**Answer**: Issues originating from the same software repository share coding idioms, developers, and issue templates, violating the IID assumption. Cluster bootstrap resamples entire repository clusters with replacement, preserving intra-repository correlations.

### Q33: Does naive IID bootstrap always underestimate uncertainty?
**Answer**: No. Our empirical audit revealed that the cluster bootstrap standard error ratio is **model-specific**:
- Random Forest: Cluster SE = 0.0349 vs IID SE = 0.0310 (**$1.13\times$ inflation**, +13%)
- Linear SVM: Cluster SE = 0.0355 vs IID SE = 0.0341 (**$1.04\times$ inflation**, +4%)
- Logistic Regression: Cluster SE = 0.0303 vs IID SE = 0.0308 (**$0.98\times$ ratio**, -2%)
Claiming that cluster bootstrap "always inflates SE by 30–40%" is empirically false.

### Q34: What bootstrap parameters were used?
**Answer**: $B=1,000$ resamples with replacement, seed 42, evaluated on test partition Seed 42 ($N=221$).

### Q35: What are the 95% cluster confidence intervals for Macro-F1?
**Answer**:
- Random Forest: $[0.6127, 0.7495]$
- Logistic Regression: $[0.6135, 0.7323]$
- Linear SVM: $[0.5151, 0.6543]$

---

## Part 9: Explainability, Attribution Stability & MDI Mechanics

### Q36: What did your explainability stability experiment test?
**Answer**: We tracked the linear model coefficients of 10 curated domain tokens between the Random Split model (subject to repository overlap) and the Repository-Independent model (zero repository overlap).

### Q37: What were the results of the 10-token test?
**Answer**: 9 out of 10 tokens (90%) maintained their directional sign:
- Flaky indicators (`flaky`, `timeout`, `race`, `intermittent`, `sleep`, `wait`, `error`) stayed strictly positive ($+$).
- Maintenance indicators (`fix`, `add`) stayed strictly negative ($-$).
- The Go-specific token `goroutine` **inverted sign** from $-0.1708$ to $+0.0387$, providing direct proof that repository clustering can alter token polarity for domain-specific keywords.

### Q38: Can Random Forest feature importances have negative signs?
**Answer**: No. Random Forest Mean Decrease in Impurity (Gini importance) is mathematically non-negative:
$$\text{MDI}(j) = \sum_{T} \sum_{t: v(t)=j} \Delta \text{Gini}(t) \ge 0$$
MDI measures how frequently a feature purifies tree nodes, not directional polarity. Claiming that Random Forest features have "positive signs" is a mathematical impossibility. Signed instance-level attributions require SHAP or LIME.

### Q39: Do token attributions prove causal mechanisms of failure?
**Answer**: No. Feature attributions reflect **statistical associations in the text representation space**. A positive weight on `race` indicates that developer issues mentioning "race" are correlated with flaky labels; it does not prove that an operating system race condition caused the hardware test failure.

---

## Part 10: Scientific Novelty, Practical Deployment & Limitations

### Q40: What is genuinely novel about this research?
**Answer**:
1. It is the first systematic evaluation of cross-repository and cross-language generalization bounds on the Barbosa et al. (*TSE 2023*) benchmark.
2. It documents the inductive bias trade-off: non-linear tree ensembles dominate in-distribution, but linear models dominate under repository and language shift.
3. It demonstrates empirical proof of feature attribution instability (`goroutine` sign inversion) under repository distribution shift.
4. It establishes rigorous empirical Monte Carlo Macro-F1 baselines for root-cause classification.

### Q41: Can FlakyGuard AI be deployed on a completely new GitHub repository?
**Answer**: Yes. The CLI and inference engine accept raw text strings and execute in $<15\text{ms}$. When applied to an unseen repository, users can expect a Macro-F1 of approximately $0.6670 \pm 0.0239$ using Logistic Regression.

### Q42: What happens if an issue description contains no text or is uninformative?
**Answer**: The inference engine implements graceful fallback tokenization (`"test failure"`), assigning baseline prior probabilities without throwing runtime errors.

### Q43: What is the main reason 13-class root-cause classification achieves only 0.2834 Macro-F1?
**Answer**: Extreme sample sparsity. Rare classes like `Test Suite Timeout` ($N=5$) and `Too Restrictive Range` ($N=6$) have only 1 sample in the test set. Grouping into 4 macro categories increases Macro-F1 to 0.4222 by absorbing within-domain classification errors and increasing training density per class.

### Q44: What are the three primary future research directions?
**Answer**:
1. **Multi-Modal CI Integration**: Combining issue text with raw runtime console output and stack traces.
2. **Context-Aware Embeddings**: Evaluating small code-specialized foundation models (e.g. CodeBERT) under strict repository grouping.
3. **Automated Triage Actioning**: Connecting root-cause predictions to automated test retry policies and quarantine triggers in CI systems.
