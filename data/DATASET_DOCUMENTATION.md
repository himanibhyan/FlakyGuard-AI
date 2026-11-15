# Dataset Documentation: TSE22 Test Flakiness Across Programming Languages

## 1. Provenance & Paper Context
- **Title**: *Test Flakiness Across Programming Languages*
- **Authors**: Keila Barbosa, Ronivaldo Ferreira, Gustavo Pinto, Marcelo d'Amorim, Breno Miranda
- **Journal**: IEEE Transactions on Software Engineering (TSE), Vol. 49, No. 4, April 2023
- **Official Repository**: [https://github.com/Test-Flaky/TSE22](https://github.com/Test-Flaky/TSE22)
- **Primary Raw Data File**: `DataBase_GitHub.xlsx` (Sheet: `Data`)

## 2. Dataset Overview
The dataset contains **1,543** GitHub issues collected across five major programming languages:
- **C** (263 issues)
- **Go** (300 issues)
- **Java** (300 issues)
- **JavaScript / js** (380 issues)
- **Python** (300 issues)

GitHub issues were initially queried using the criteria:
1. Closed issues with label `bug`
2. Containing keywords `"flaky"` and `"test"`
3. Selected across active open-source repositories in C, Go, Java, JavaScript, Python

## 3. Schema & Column Specifications
The sheet `Data` in `DataBase_GitHub.xlsx` has header annotations on rows 0-1 and formal field names on row 2:

| Raw Column Name | Standardized Name | Type | Description | Non-Null Count | Missing % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ID` | `issue_id` | Integer | Unique identifier for each issue | 1543 | 0.0% |
| `Repository/Project` | `repository` | String | GitHub repository name | 1351 | 12.4% |
| `Language` | `language` | String | Programming language (`go`, `python`, `java`, `js`, `C`) | 1543 | 0.0% |
| `Status` | `flakiness_status` | String | Ground-truth flakiness label: `T` (True), `F` (False positive), `ND` (Not determined), `DUPLICATE`, `NaN` | 1353 | 12.3% |
| `Labels` | `labels` | String | Labels assigned on the GitHub issue | 1343 | 13.0% |
| `Status Issue` | `issue_status` | String | State on GitHub (e.g., `closed`, `merged`) | 1341 | 13.1% |
| `Year` | `year` | Numeric | Year issue was created (2012–2021) | 1337 | 13.4% |
| `Category` (col 7) | `root_cause_category` | String | Author-assigned root-cause problem category | 1353 | 12.3% |
| `Reviewers` (col 8) | `root_cause_reviewers`| String | Paper authors who reviewed the root-cause label (`A,B`) | 1353 | 12.3% |
| `Description` (col 9) | `problem_description_excerpt` | String | Excerpt from issue text supporting root-cause category | 1341 | 13.1% |
| `URL Issue` | `url_issue` | String | Canonical URL to GitHub issue/PR | 1543 | 0.0% |
| `Category.1` (col 11)| `fix_strategy_category`| String | Author-assigned fix strategy category | 1447 | 6.2% |
| `Reviewers.1` | `fix_strategy_reviewers`| String | Paper authors reviewing fix strategy | 1353 | 12.3% |
| `Description.1` | `fix_description_excerpt` | String | Excerpt from issue text supporting fix strategy | 1340 | 13.2% |
| `M1...` | `days_to_close` | Numeric | Metric M1: Number of days until issue was closed | 1348 | 12.6% |
| `M2...` | `comments_count` | Numeric | Metric M2: Number of comments on the issue | 1346 | 12.8% |
| `Domain` | `domain` | String | Application domain (Network, File, GUI, Web, etc.) | 1345 | 12.8% |

### Enriched Issue Report Fields
Because the raw spreadsheet contains only short reviewer excerpts rather than the full issue report, our acquisition pipeline scraped and cached the authentic GitHub issue metadata:
- `issue_title`: Official title of the GitHub issue / pull request
- `issue_body`: Full initial text description submitted by the issue creator

## 4. Class Distributions & Statistical Audits

### Flakiness Status (`flakiness_status`)
- `T` (True Flaky): **703** (45.6%)
- `F` (False Positive): **401** (26.0%)
- `ND` (Not Determined): **224** (14.5%)
- `NaN` (Uninspected): **190** (12.3%)
- `DUPLICATE`: **25** (1.6%)

For Task A (Binary Flakiness Detection), `ND`, `DUPLICATE`, and uninspected records are excluded, leaving **1,104** clean labeled instances (703 `T` vs 401 `F`). This represents an imbalanced ratio of ~63.7% True to 36.3% False Positive.

### Root-Cause Categories (`root_cause_category`)
Total annotated records with confirmed root causes (excluding `--` and `Hard to Classify`):
1. **Async Wait**: 103 (17.4%)
2. **Concurrency**: 97 (16.4%)
3. **Platform Dependency**: 87 (14.7%)
4. **Network**: 72 (12.2%)
5. **Resource Leak**: 61 (10.3%)
6. **Test Order Dependency**: 40 (6.8%)
7. **Test Case Timeout**: 28 (4.7%)
8. **Time**: 28 (4.7%)
9. **Unordered Collections**: 28 (4.7%)
10. **Randomness**: 22 (3.7%)
11. **Floating Point Operations**: 14 (2.4%)
12. **Too Restrictive Range**: 6 (1.0%)
13. **Test Suite Timeout**: 5 (0.8%)

*Note*: 335 instances are labeled `Hard to Classify` and 422 instances have `--` (unannotated or non-flaky).

### Single-Label vs Multi-Label Verification
- `root_cause_category`: Analysis confirmed each issue has a **single discrete category** assigned by researchers.
- `fix_strategy_category`: While some fix strategy codes have compound abbreviations (e.g. `AW - Add/modify waitFor`), each row is mapped to a primary fix category.

## 5. Critical Data Integrity & Leakage Prevention Rules
1. **No Target Leakage in Task A**: Reviewer notes (`problem_description_excerpt`, `fix_description_excerpt`), resolution strategies, and root-cause labels are strictly **omitted** from model input features during Flakiness Detection.
2. **Text Representation**:
   - Primary feature: `combined_text = issue_title + " " + issue_description`
   - Preprocessing includes lowercase normalization, removal of markdown artifacts/URLs, handling of code blocks, and tokenization.
3. **Exclusion of Indeterminate Records**: All records with status `ND` (Not Determined) or unannotated `NaN` are pruned from binary classification to prevent label noise.
