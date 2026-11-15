# FlakyGuard AI: Critical Dataset Reconciliation & Filtering Audit Report

**Date**: November 15, 2026  
**Auditor Role**: Independent Senior Machine Learning Researcher & Methodology Auditor  
**Primary Data Source**: `data/raw/DataBase_GitHub.xlsx` (Sheet: `Data`, Sheet: `ProjectsDomain`)  
**Processed Datasets**: `data/processed/flaky_binary_dataset.csv`, `data/processed/root_cause_dataset.csv`, `data/processed/full_cleaned_dataset.csv`  

---

## 1. Executive Summary

This independent investigation inspected every sheet, row, and column of the raw Excel benchmark from Barbosa et al. (*IEEE TSE 2023*) and verified every record transformation in the preprocessing pipeline. 

All previously reported sample counts, label subsets, and repository numbers have been verified against the raw files:
- **Raw Issue Records**: Exactly **1,543** rows in sheet `Data` (header on row 2, rows 0–1 are author notes).
- **Deduplicated Issues**: Exactly **1,513** issues after removing 25 explicitly labeled `DUPLICATE` records and 5 redundant `URL Issue` rows.
- **Binary Classification Dataset**: Exactly **1,101** issues (**701 Flaky `T`**, **400 False Positive `F`**), after excluding 222 indeterminate `ND` and 190 uninspected/null records.
- **Root-Cause Dataset**: Exactly **589** confirmed flaky issues across **13 discrete categories**, after excluding 112 uninformative/hard-to-classify issues from the 701 `T` issues.
- **Repository Count Resolution**:
  - Raw Excel contains **850 canonical repositories**.
  - Deduplicated dataset contains **842 canonical repositories**.
  - Binary dataset contains **641 canonical repositories**.
  - Root-cause dataset contains **349 canonical repositories**.
  - The figure **421** originates from row 2 of the author's metadata tab `ProjectsDomain` ("Nº of Projects with at least 1 Flaky") and is an author domain-analysis summary figure, **not** the unique repository count of the binary classification dataset.

---

## 2. Raw Dataset Inspection (`data/raw/DataBase_GitHub.xlsx`)

The raw Excel workbook contains 9 distinct worksheets:
1. `Data` (1,545 rows total; rows 0-1 are explanatory notes; row 2 is the formal header; 1,543 data rows, 17 columns).
2. `Problem` (16 rows, 8 columns).
3. `Solution` (33 rows, 10 columns).
4. `Table 4 and 5` (96 rows, 13 columns).
5. `Table 6, 7, 8 and 9 ` (121 rows, 19 columns).
6. `Table10` (105 rows, 15 columns).
7. `Table 11` (85 rows, 24 columns).
8. `IssueNonFlaky` (5,005 rows, 4 columns).
9. `ProjectsDomain` (31 rows, 11 columns).

### Column Specifications and Missing Values in Sheet `Data` ($N=1,543$):

| Column Index | Raw Header | Missing Values | % Missing | Standardized Field Name |
| :---: | :--- | :---: | :---: | :--- |
| 0 | `ID` | 0 | 0.0% | `issue_id` |
| 1 | `Repository/Project` | 192 | 12.44% | `repository` (fallback to URL regex) |
| 2 | ` Language` | 0 | 0.0% | `language` |
| 3 | `Status` | 190 | 12.31% | `flakiness_status` |
| 4 | `Labels` | 200 | 12.96% | `labels` |
| 5 | `Status Issue` | 202 | 13.09% | `issue_status` |
| 6 | `Year` | 206 | 13.35% | `year` |
| 7 | `Category` | 190 | 12.31% | `root_cause_category` |
| 8 | `Reviewers` | 190 | 12.31% | `root_cause_reviewers` |
| 9 | `Description` | 202 | 13.09% | `problem_description_excerpt` |
| 10 | `URL Issue` | 0 | 0.0% | `url_issue` (100% complete) |
| 11 | `Category.1` | 96 | 6.22% | `fix_strategy_category` |
| 12 | `Reviewers.1` | 190 | 12.31% | `fix_strategy_reviewers` |
| 13 | `Description.1` | 203 | 13.16% | `fix_description_excerpt` |
| 14 | `M1\nNumber of days...` | 195 | 12.64% | `days_to_close` |
| 15 | `M2\nNumber of comments...`| 197 | 12.77% | `comments_count` |
| 16 | `Domain` | 198 | 12.83% | `domain` |

---

## 3. Record-Level Filtering Traceability

The table below traces the exact filtering arithmetic from raw records to model-ready datasets:

| Filtering Stage | Input Count | Removed Records | Output Count | Primary Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Dataset (Sheet: `Data`)** | 1,545 | 2 (Header notes) | **1,543** | Rows 0-1 are annotation text; row 2 is column header |
| **Deduplication Stage 1** | 1,543 | 25 | 1,518 | Explicitly marked `Status == "DUPLICATE"` in raw spreadsheet |
| **Deduplication Stage 2** | 1,518 | 5 | **1,513** | Redundant duplicate issue URLs (`URL Issue` duplicate rows) |
| **Binary Label Validation** | 1,513 | 412 | **1,101** | Excluded 222 `ND` (Not Determined) and 190 uninspected/null records |
| **Task A: Positive Class (Flaky)** | 1,101 | 400 | **701** | Confirmed flaky test issues (`flakiness_status == "T"`, 63.67%) |
| **Task A: Negative Class (False Pos)**| 1,101 | 701 | **400** | Confirmed false positives / regular bugs (`flakiness_status == "F"`, 36.33%) |
| **Root-Cause Filtering** | 701 | 112 | **589** | Excluded 111 `Hard to Classify` and 1 `Hard to classify` from `T` issues |
| **Task B: 13-Class Taxonomy** | 589 | 0 | **589** | Confirmed flaky issues with discrete failure mode annotations |

---

## 4. Investigation of the 421 vs. 641 Repository Discrepancy

A major point of confusion in earlier research notes was the coexistence of "421 repositories" and "641 repositories". This audit inspected the raw spreadsheet to determine the exact origin of both figures:

### Origin of "421 Projects":
In `DataBase_GitHub.xlsx`, sheet `ProjectsDomain`:
- Row 0: Language names (`C`, `Go`, `Java`, `Javascript`, `Python`, `Total`).
- Row 1: `Nº of Analyzed Projects (includes T, ND or F)`: C=121, Go=107, Java=126, JS=248, Python=158, **Total = 760**.
- Row 2: `Nº of Projects with at least 1 Flaky (Includes Problem/Solution issues Hard to classify)`: C=64, Go=74, Java=86, JS=106, Python=91, **Total = 421**.
- Row 3: `Nº of Projects with at least 1 Flaky (Issues with classified sources-fixes)`: C=52, Go=55, Java=58, JS=89, Python=68, **Total = 322**.

**Finding**: The number **421** is the original authors' project filtering count for their domain and metric analysis in `ProjectsDomain`. It defines projects containing at least one flaky issue under the authors' specific selection rules.

### Origin of "641 Canonical Repositories":
In the actual binary classification dataset ($N=1,101$), every issue contains a complete GitHub URL (`URL Issue`). When parsed with `parse_canonical_repo(url)` using regex `github.com/{owner}/{repo}`:
- **Unique Canonical Repositories = 641**.
- In the training/test split across 10 seeds (`GroupShuffleSplit`), exactly $512 \pm 3$ repositories are in training and $129 \pm 3$ repositories are in testing, strictly summing to **641**.

### Origin of "349 Root-Cause Repositories":
In the root-cause dataset ($N=589$ confirmed flaky issues), applying `parse_canonical_repo(url)` yields exactly **349 unique canonical repositories**.

**Conclusion**: The numbers 421, 641, and 349 represent distinct scopes and must never be used interchangeably:
- **421**: Author metadata project count in sheet `ProjectsDomain` (Row 2).
- **641**: Actual canonical repositories in the 1,101-issue binary dataset.
- **349**: Actual canonical repositories in the 589-issue root-cause dataset.

---

## 5. Programming Language Distribution

| Programming Language | Raw Excel ($N=1,543$) | Deduplicated ($N=1,513$) | Binary Dataset ($N=1,101$) | Root-Cause Dataset ($N=589$) |
| :--- | :---: | :---: | :---: | :---: |
| **JavaScript (`js`)** | 380 | 370 | 265 | 158 |
| **Python (`python`)** | 300 | 296 | 222 | 118 |
| **Java (`java`)** | 300 | 293 | 211 | 112 |
| **C (`C`)** | 263 | 262 | 211 | 108 |
| **Go (`go`)** | 300 | 292 | 192 | 93 |
| **Total** | **1,543** | **1,513** | **1,101** | **589** |

---

## 6. Root-Cause Distribution Across 13 Categories ($N=589$)

| Rank | Root-Cause Category | Sample Count ($N=589$) | Class Percentage | Macro Category (4 Classes) |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Async Wait** | 103 | 17.49% | Timing & Asynchrony |
| 2 | **Concurrency** | 97 | 16.47% | Concurrency & Non-Determinism |
| 3 | **Platform Dependency** | 86 | 14.60% | Resource & Environment |
| 4 | **Network** | 71 | 12.05% | Resource & Environment |
| 5 | **Resource Leak** | 61 | 10.36% | Resource & Environment |
| 6 | **Test Order Dependency** | 40 | 6.79% | Test Logic & Assertions |
| 7 | **Test Case Timeout** | 28 | 4.75% | Timing & Asynchrony |
| 8 | **Time** | 28 | 4.75% | Timing & Asynchrony |
| 9 | **Unordered Collections** | 28 | 4.75% | Concurrency & Non-Determinism |
| 10 | **Randomness** | 22 | 3.74% | Concurrency & Non-Determinism |
| 11 | **Floating Point Operations** | 14 | 2.38% | Test Logic & Assertions |
| 12 | **Too Restrictive Range** | 6 | 1.02% | Test Logic & Assertions |
| 13 | **Test Suite Timeout** | 5 | 0.85% | Timing & Asynchrony |
| — | **Total** | **589** | **100.00%** | **4 Categories** |
