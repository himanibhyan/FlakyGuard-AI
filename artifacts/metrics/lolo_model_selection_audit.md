# FlakyGuard AI: Leave-One-Language-Out (LOLO) Model Selection Bias Audit

**Audit Date**: November 2, 2026  
**Auditor**: Senior Research Validation Engineer  
**Objective**: Detect, analyze, and rectify post-hoc model selection bias in cross-language generalization evaluation.

---

## 1. Methodological Critique: Post-Hoc Test Selection Bias

In preliminary cross-language reporting, a summary table presented a "Best Architecture" for each held-out language (selecting Random Forest for JavaScript and Go, while selecting Logistic Regression for C and Java).

### Why This Was Methodologically Biased:
1. **Test-Set Snooping (Data Leakage)**: Selecting the "best" model for a held-out language based on which model achieved the highest test-set metric violates the fundamental separation between model selection and evaluation.
2. **Unrealistic Deployment Assumption**: In a true continuous integration scenario where a new programming language is introduced to a code intelligence tool, developers have **zero labeled examples** for that language. One cannot select a model based on unobserved test performance.
3. **Rectified Protocol**:
   - The test set for each held-out language must remain strictly quarantined from model selection.
   - All three candidate architectures (Random Forest, Logistic Regression, Linear SVM) must be evaluated across all five languages.
   - Cross-language transferability must be measured using the complete, unselected factorial matrix ($5 \times 3 = 15$ evaluations).
   - Any overall model preference must be established either *a priori* or via aggregate cross-language performance.

---

## 2. Complete, Unbiased Factorial LOLO Evaluation Matrix

The table below presents the results for all three models across all five held-out languages. Preprocessing and TF-IDF vectorization were fitted strictly on the four training languages in each fold:

| Held-Out Language | Architecture | Train N | Test N | Flaky Ratio | Accuracy | Precision | Recall | F1-Score | Macro-F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C** | **Random Forest** | 890 | 211 | 56.87% | 0.6019 | 0.8000 | 0.4000 | 0.5333 | 0.5931 | 0.6827 | 0.7626 |
| **C** | **Logistic Regression**| 890 | 211 | 56.87% | 0.6682 | 0.8788 | 0.4833 | 0.6237 | **0.6635** | 0.7259 | 0.8077 |
| **C** | **Linear SVM** | 890 | 211 | 56.87% | 0.5972 | 0.5946 | 0.9167 | 0.7213 | 0.4974 | 0.7271 | 0.8136 |
| **Go** | **Random Forest** | 909 | 192 | 78.12% | 0.7448 | 0.8389 | 0.8333 | 0.8361 | 0.6298 | 0.7092 | 0.8761 |
| **Go** | **Logistic Regression**| 909 | 192 | 78.12% | 0.7344 | 0.8462 | 0.8067 | 0.8259 | **0.6327** | 0.7157 | 0.8918 |
| **Go** | **Linear SVM** | 909 | 192 | 78.12% | 0.7656 | 0.7966 | 0.9400 | 0.8624 | 0.5365 | 0.6937 | 0.8802 |
| **Java** | **Random Forest** | 890 | 211 | 75.36% | 0.7488 | 0.7624 | 0.9686 | 0.8532 | 0.4922 | 0.5192 | 0.7700 |
| **Java** | **Logistic Regression**| 890 | 211 | 75.36% | 0.7251 | 0.7644 | 0.9182 | 0.8343 | **0.5144** | 0.4874 | 0.7320 |
| **Java** | **Linear SVM** | 890 | 211 | 75.36% | 0.7630 | 0.7608 | 1.0000 | 0.8641 | 0.4691 | 0.4878 | 0.7341 |
| **JavaScript** | **Random Forest** | 836 | 265 | 50.94% | 0.6943 | 0.6824 | 0.7481 | 0.7138 | **0.6929** | 0.7636 | 0.7470 |
| **JavaScript** | **Logistic Regression**| 836 | 265 | 50.94% | 0.6868 | 0.7167 | 0.6370 | 0.6745 | 0.6863 | 0.7738 | 0.7579 |
| **JavaScript** | **Linear SVM** | 836 | 265 | 50.94% | 0.5925 | 0.5590 | 0.9481 | 0.7033 | 0.5263 | 0.7563 | 0.7394 |
| **Python** | **Random Forest** | 879 | 222 | 61.71% | 0.6757 | 0.6757 | 0.9124 | 0.7764 | **0.5931** | 0.6000 | 0.6711 |
| **Python** | **Logistic Regression**| 879 | 222 | 61.71% | 0.5315 | 0.6947 | 0.4818 | 0.5593 | 0.5297 | 0.5403 | 0.6440 |
| **Python** | **Linear SVM** | 879 | 222 | 61.71% | 0.6171 | 0.6171 | 1.0000 | 0.7632 | 0.3816 | 0.5745 | 0.6495 |

---

## 3. Aggregate Performance Across Languages (Without Post-Hoc Selection)

When evaluating each architecture across all five held-out languages without cherry-picking:

| Architecture | Mean Macro-F1 | Std Macro-F1 | Min Macro-F1 | Max Macro-F1 | Mean Accuracy | Mean ROC-AUC | Mean PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.6053** | $\pm 0.0818$ | 0.5144 (Java) | 0.6863 (JS) | 0.6692 | 0.6486 | 0.7667 |
| **Random Forest** | **0.6002** | $\pm 0.0722$ | 0.4922 (Java) | 0.6929 (JS) | **0.6931** | **0.6549** | **0.7654** |
| **Linear SVM** | **0.4822** | $\pm 0.0610$ | 0.3816 (Python) | 0.5365 (Go) | 0.6671 | 0.6479 | 0.7634 |

### Critical Comparative Insights:
1. **Logistic Regression Wins on Out-of-Distribution Macro-F1**:
   While Random Forest was the superior model on the in-distribution random split (Macro-F1: 0.6848 vs. 0.6555), **Logistic Regression achieves the highest mean Macro-F1 under cross-language distribution shift (0.6053 vs. 0.6002)**.
   *Reason*: Linear models assign global weights across the entire feature space, whereas decision trees construct hard orthogonal splits that can fail abruptly when language-specific tokens are absent.
2. **Linear SVM Vulnerability to Threshold Shift**:
   Linear SVM exhibited a severe pathology: across Java, Python, and Go, it collapsed into predicting almost all instances as Flaky (Recall = 94% to 100%). Because the decision threshold cannot adapt to shifting base rates in unseen languages, its Macro-F1 drops to 0.4822.
3. **Transferability Conclusion**:
   Both Random Forest and Logistic Regression provide viable zero-shot cross-language transfer (Macro-F1 ~0.60-0.69 on JavaScript, Go, and C), but cross-language performance is strictly lower than in-distribution performance (Macro-F1 ~0.71).
