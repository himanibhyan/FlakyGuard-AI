# FlakyGuard AI: Dataset Count Reconciliation & Repository Census

**Audit Date**: November 2, 2026  
**Auditor**: Senior Research Validation Engineer  
**Objective**: Reconcile the discrepancy between the historical literature citation of "421 repositories" and the empirical census of "641 canonical repositories" observed during repository-independent evaluation.

---

## 1. Executive Reconciliation Summary

The apparent discrepancy between the citation of **421 repositories** and the **641 unique repositories** partitioned in the repository-independent experiment is caused by differing scope definitions:

1. **Origin of the "421" Figure**:
   In the official TSE22 benchmark spreadsheet (`DataBase_GitHub.xlsx`, sheet `ProjectsDomain`, row 2), the researchers explicitly define:
   $$\text{"Nº of Projects with at least 1 Flaky (Includes Problem/Solution issues Hard to classify)"} = \mathbf{421}$$
   The figure **421 refers exclusively to the subset of GitHub projects that contained at least one confirmed flaky issue (`T`)**. It was never the total number of projects in the dataset. In row 1 of the same sheet, the authors report:
   $$\text{"Nº of Analyzed Projects (includes T, ND or F)"} = \mathbf{760}$$

2. **Origin of the "641" Figure**:
   In the binary flakiness dataset ($N=1,101$), both true flaky tests (`T`: 701) and false positive issues (`F`: 400) are evaluated. When repository names are canonicalized to unique GitHub `{owner}/{repo}` identifiers parsed from `url_issue`, these 1,101 issues map to exactly **641 unique canonical repositories**.

3. **Partitioning Arithmetic**:
   $$\text{Train Repositories } (512) + \text{Test Repositories } (129) = \mathbf{641 \text{ Unique Repositories}} \quad (\text{Overlap} = 0)$$

---

## 2. Comprehensive Census Across All Pipeline Stages

The following table presents actual computed counts at each stage of data preparation, verifying both raw spreadsheet strings and canonical URL-derived repository identifiers:

| Dataset Stage | Issue Count | Raw `repository` Column Strings | Canonical `{owner}/{repo}` Count | Scope Description & Filtering Criteria |
| :--- | :---: | :---: | :---: | :--- |
| **Original Spreadsheet (`DataBase_GitHub.xlsx`)** | **1,543** | 761 (192 nulls) | **850** | All extracted issues across C, Go, Java, JavaScript, Python. |
| **After Deduplication (`full_cleaned_dataset.csv`)** | **1,513** | 758 (188 nulls) | **842** | Removed 30 duplicate URLs and duplicate flags. |
| **Binary Classification Dataset (`flaky_binary_dataset.csv`)** | **1,101** | 653 (1 null) | **641** | Labeled subset: `T` (701) and `F` (400). Excluded `ND` (224). |
| **Root-Cause Dataset (`root_cause_dataset.csv`)** | **589** | 363 (0 nulls) | **349** | Confirmed flaky tests (`T`) across 13 verified categories. |
| **Repository-Independent Train Set** | **874** | — | **512** | 80% group partition via `GroupShuffleSplit(seed=42)`. |
| **Repository-Independent Test Set** | **227** | — | **129** | 20% group partition via `GroupShuffleSplit(seed=42)`. |

*Verification Check*: $512 \text{ (Train)} + 129 \text{ (Test)} = 641 \text{ (Total Canonical Repositories)}$. Intersecting set size $= 0$.

---

## 3. Discrepancy Breakdown: Raw Column vs. Canonical URL Parsing

In the raw spreadsheet `Data` tab, the column `Repository/Project` contains data-entry inconsistencies introduced during manual annotation:
1. **Missing Owner Names**: Annotators recorded repository names without their GitHub organization/owner prefix (e.g., recording `terraform` instead of `hashicorp/terraform`, or `quic-go` instead of `lucas-clemente/quic-go`).
2. **Missing Repository Values**: 192 rows in the raw sheet had an empty `Repository/Project` cell, but every single row contained a valid `URL Issue` link (e.g., `https://github.com/DataDog/dd-trace-py/pull/193`).
3. **Repository Name Synonyms**: Different annotators occasionally wrote alternate shorthand strings for the same underlying codebase (e.g., `pandas` vs. `pandas-dev`, or `mbedtls` vs. `mbed-tls`).
4. **Canonicalization Resolution**:
   Our canonical extractor:
   ```python
   def parse_canonical_repo(url: str, fallback_repo: str = "") -> str:
       m = re.search(r"github\.com/([^/]+)/([^/]+)/", str(url))
       if m:
           return f"{m.group(1)}/{m.group(2)}"
       return fallback_repo.strip()
   ```
   resolves each issue to its authoritative `{owner}/{repo}` slug, completely eliminating missing values and preventing cross-repository entity collisions.

---

## 4. Reconciled Statistical Summary

- The literature citation of **421 repositories** is valid only as a historical count of projects possessing confirmed flakiness annotations.
- The operational population of the binary classification task comprises **1,101 issues across 641 canonical repositories**.
- The repository-independent split strictly segregates these 641 projects into **512 training repositories and 129 testing repositories with zero leakage**.
