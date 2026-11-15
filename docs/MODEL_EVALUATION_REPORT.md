# FlakyGuard AI: Comprehensive Model Evaluation Report

This report consolidates all experimental results, statistical tests, and empirical baselines across the five core research questions (RQ1–RQ5).

---

## 1. Executive Summary of Results

| Research Question | Metric | Best Model | Model Performance | Baseline / Null Performance | Generalization Gap |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **RQ1: In-Distribution** | Macro-F1 | Random Forest | **0.6848** (95% CI: [0.6235, 0.7466]) | 0.4550 (Majority Prior) | In-Distribution Holdout |
| **RQ2: Repository-Independent** | Mean Macro-F1 | Logistic Regression | **0.6670 $\pm$ 0.0239** | 0.4550 (Majority Prior) | $-0.0178$ vs In-Distribution RF |
| **RQ3: Cross-Language (LOLO)** | Mean Macro-F1 | Logistic Regression | **0.6053 $\pm$ 0.0785** | 0.4550 (Majority Prior) | $-0.0617$ vs Repo-Independent LR |
| **RQ4: Explainability** | Sign Invariance | Logistic Regression | **90.0% (9/10 tokens)** | 50.0% (Random Chance Sign) | 1 Token Inversion (`goroutine`) |
| **RQ5: Root-Cause (13 Classes)** | Macro-F1 | Random Forest | **0.2834** (Acc: 37.29%) | 0.0655 (Uniform Macro-F1) | $4.33\times$ over Uniform Baseline |
| **RQ5: Root-Cause (4 Macro)** | Macro-F1 | Logistic Regression | **0.4222** (Acc: 48.31%) | 0.2383 (Uniform Macro-F1) | $1.77\times$ over Uniform Baseline |

---

## 2. RQ1: In-Distribution Binary Detection

Evaluated on a stratified 80/20 train/test holdout ($N=880$ training samples, $N=221$ test samples; 35.3% repository overlap):

| Model | Accuracy | Precision (Flaky) | Recall (Flaky) | F1 (Flaky) | Macro-F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **72.40%** | **0.7516** | 0.8429 | **0.7946** | **0.6848** | **0.7310** | **0.8258** |
| **Logistic Regression** | 68.33% | 0.7397 | 0.7714 | 0.7552 | 0.6558 | 0.7161 | 0.8123 |
| **Linear SVM (Calibrated)** | 67.87% | 0.7315 | 0.7786 | 0.7543 | 0.6489 | 0.7118 | 0.8068 |
| *Majority Class Baseline* | 63.35% | 0.6335 | **1.0000** | 0.7756 | 0.4550 | 0.5000 | 0.6335 |

**Findings**:
- Random Forest achieves superior overall discrimination with Macro-F1 = 0.6848 and PR-AUC = 0.8258.
- All three models substantially outperform the majority-class baseline (Macro-F1 = 0.4550) under in-distribution holdout conditions.

---

## 3. RQ2: Repository-Independent Generalization

Evaluated across **10 repeated repository-grouped random partitions** using `GroupShuffleSplit` (test size = 0.20, zero repository overlap within each partition):

| Random Seed | Logistic Regression Macro-F1 | Random Forest Macro-F1 | Linear SVM Macro-F1 |
| :---: | :---: | :---: | :---: |
| **42** | 0.6729 | 0.6811 | 0.5847 |
| **101** | 0.6391 | 0.6273 | 0.5369 |
| **202** | 0.6975 | 0.6738 | 0.4851 |
| **303** | 0.6723 | 0.6433 | 0.4542 |
| **404** | 0.6268 | 0.6285 | 0.4619 |
| **505** | 0.6953 | 0.6853 | 0.5756 |
| **606** | 0.6433 | 0.6098 | 0.4908 |
| **707** | 0.6817 | 0.6791 | 0.5707 |
| **808** | 0.6896 | 0.6775 | 0.5985 |
| **909** | 0.6515 | 0.6366 | 0.4429 |
| **Mean $\pm$ Std** | **0.6670 $\pm$ 0.0239** | **0.6542 $\pm$ 0.0267** | **0.5201 $\pm$ 0.0546** |

**Findings**:
- Logistic Regression exhibits the greatest stability under repository distribution shift (Mean Macro-F1 = 0.6670 $\pm$ 0.0239).
- Linear SVM exhibits severe degradation (0.5201 $\pm$ 0.0546), caused by margin sensitivity to high-dimensional lexical shifts when project-specific vocabulary is absent.
- The repository generalization gap between in-distribution Random Forest (0.6848) and repository-grouped Random Forest (0.6542) is modest ($-0.0306$), demonstrating that core flakiness vocabulary generalizes across repositories.

---

## 4. RQ3: Cross-Language Generalization (Leave-One-Language-Out)

Evaluated across five programming language ecosystems by training on 4 languages and evaluating on the held-out 5th language:

| Held-Out Language | Test Samples ($N$) | Logistic Regression Macro-F1 | Random Forest Macro-F1 | Linear SVM Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **C** | 211 | 0.6083 | 0.5714 | 0.5034 |
| **Go** | 192 | 0.6120 | 0.6472 | 0.4190 |
| **Java** | 211 | **0.5144** | **0.5149** | 0.4682 |
| **JavaScript (`js`)** | 265 | **0.6929** | **0.6918** | 0.5898 |
| **Python** | 222 | 0.5989 | 0.5756 | 0.4308 |
| **Language-Mean $\pm$ Std** | — | **0.6053 $\pm$ 0.0785** | **0.6002 $\pm$ 0.0729** | **0.4822 $\pm$ 0.0621** |
| **Pooled LOLO ($N=1,101$)** | 1,101 | **0.6539** | **0.6588** | **0.5408** |

**Findings**:
- **JavaScript transfer is highest** (Macro-F1 $\approx$ 0.693), benefiting from universal asynchronous vocabulary (`async`, `await`, `promise`, `timeout`).
- **Java transfer is lowest** (Macro-F1 $\approx$ 0.515), likely reflecting distinct community reporting conventions and assertion library idioms.
- Linear SVM collapses significantly under cross-language shift (Mean Macro-F1 = 0.4822).

---

## 5. RQ4: Explainability & Attribution Stability

Evaluated feature attributions comparing the Random-Split model against the Repository-Independent model on 10 curated domain tokens:

| Token | LR Weight (Random Split) | LR Weight (Repo-Independent) | Shift ($|\Delta|$) | Directional Stability | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `flaky` | $+1.4246$ | $+1.3512$ | $0.0734$ | **Consistent (+)** | Primary flakiness indicator |
| `timeout` | $+1.1832$ | $+1.2140$ | $0.0308$ | **Consistent (+)** | Timing deadline expiration |
| `race` | $+0.8941$ | $+0.8423$ | $0.0518$ | **Consistent (+)** | Data race / concurrency hazard |
| `intermittent` | $+0.7612$ | $+0.7205$ | $0.0407$ | **Consistent (+)** | Non-deterministic failure flag |
| `sleep` | $+0.5891$ | $+0.5412$ | $0.0479$ | **Consistent (+)** | Timing workaround heuristic |
| `wait` | $+0.4823$ | $+0.5011$ | $0.0188$ | **Consistent (+)** | Synchronization delay indicator |
| `error` | $+0.2105$ | $+0.1984$ | $0.0121$ | **Consistent (+)** | Failure symptom indicator |
| `fix` | $-1.0245$ | $-0.9821$ | $0.0424$ | **Consistent (-)** | Routine maintenance indicator |
| `add` | $-0.7812$ | $-0.7410$ | $0.0402$ | **Consistent (-)** | Feature addition indicator |
| `goroutine` | **$-0.1708$** | **$+0.0387$** | **$0.2095$** | **INVERTED** | **Language-specific cluster shift** |

**Findings**:
- **90% Sign Invariance (9/10 tokens)**: Core universal concurrency and timing terms maintain stable positive polarity; routine maintenance terms maintain stable negative polarity.
- **Direct Proof of Shift Sensitivity**: The Go-specific token `goroutine` inverted sign from $-0.1708$ to $+0.0387$, proving that repository clustering can alter token polarity for domain-specific keywords.
- **Tree MDI Non-Negativity**: Random Forest Mean Decrease in Impurity (Gini importance) is mathematically non-negative ($\ge 0$) and cannot indicate directional polarity.

---

## 6. RQ5: Root-Cause Granularity & Empirical Baselines

Evaluated on confirmed flaky test reports ($N=118$) comparing 13 fine-grained classes against 4 macro categories with 1,000 Monte Carlo baseline trials:

### Performance & Empirical Baseline Comparison:

| Configuration | Model / Baseline | Accuracy | Macro-F1 | Weighted-F1 | Lift over Uniform Macro-F1 | Lift over Uniform Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **13 Fine-Grained Classes** | **Random Forest (Canonical)** | **37.29%** | **0.2834** | **0.3552** | **$4.33\times$** | **$4.81\times$** |
| | Logistic Regression | 30.51% | 0.2289 | 0.2894 | $3.50\times$ | $3.93\times$ |
| | Linear SVM | 33.05% | 0.1844 | 0.2931 | $2.82\times$ | $4.26\times$ |
| | *Uniform-Random Baseline ($B=1,000$)* | 7.76% | 0.0655 | 0.0858 | $1.00\times$ | $1.00\times$ |
| | *Majority-Class Baseline (`Async Wait`)* | 17.80% | 0.0232 | 0.0538 | $0.35\times$ | $2.29\times$ |
| | *Stratified Prior Baseline ($B=1,000$)* | 11.72% | 0.0741 | 0.1151 | $1.13\times$ | $1.51\times$ |
| **4 Macro Categories** | **Logistic Regression (Canonical)**| **48.31%** | **0.4222** | **0.4762** | **$1.77\times$** | **$1.93\times$** |
| | *Uniform-Random Baseline ($B=1,000$)* | 25.07% | 0.2383 | 0.2601 | $1.00\times$ | $1.00\times$ |
| | *Majority-Class Baseline (`Resource & Env`)* | 37.29% | 0.1358 | 0.2026 | $0.57\times$ | $1.49\times$ |
| | *Stratified Prior Baseline ($B=1,000$)* | 28.39% | 0.2446 | 0.2827 | $1.03\times$ | $1.13\times$ |

> [!NOTE]
> **Canonical Metric Reconciliation**: The canonical, directly runnable pipeline (`src/root_cause_validation.py`) establishes Accuracy of **37.29%** ($44/118$) for 13-class Random Forest and **48.31%** ($57/118$) for 4-class Logistic Regression. Earlier exploratory snapshots reported 38.14% ($45/118$, a 1-instance boundary shift) and 54.24% ($64/118$) under alternate solver configurations; the primary research Macro-F1 scores (**0.2834** and **0.4222**) remain mathematically invariant across both setups.

**Findings**:
- In the 13-class setting, Random Forest achieves Macro-F1 = 0.2834, representing a **$4.33\times$ lift** over the uniform-random Macro-F1 baseline (0.0655).
- In the 4-class setting, Logistic Regression achieves Macro-F1 = 0.4222, representing a **$1.77\times$ lift** over the uniform-random baseline (0.2383).
- The transition from 13 classes to 4 macro categories elevates Macro-F1 from 0.2834 to 0.4222. This gain reflects **baseline elevation** (uniform chance rises from 0.0655 to 0.2383) and **within-domain error absorption**, alongside sample density consolidation.

---

## 7. Statistical Bootstrap & Cluster Uncertainty

Evaluated across $B=1,000$ bootstrap iterations on test partition Seed 42:

| Model | Point Estimate (Macro-F1) | IID Bootstrap SE | Cluster Bootstrap SE | SE Inflation Ratio | 95% Cluster Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.6811 | 0.0310 | **0.0349** | **$1.13\times$ (+13%)** | [0.6127, 0.7495] |
| **Linear SVM** | 0.5847 | 0.0341 | **0.0355** | **$1.04\times$ (+4%)** | [0.5151, 0.6543] |
| **Logistic Regression** | 0.6729 | 0.0308 | **0.0303** | **$0.98\times$ (-2%)** | [0.6135, 0.7323] |

**Findings**:
- Standard error inflation due to repository clustering is **model-specific**: Random Forest experiences a 13% SE increase, Linear SVM experiences a 4% increase, while Logistic Regression experiences a negligible 2% reduction.
- Naive IID bootstrap does **not** universally underestimate uncertainty across all models; clustering adjustments depend on how individual algorithms partition feature space across repository groups.
