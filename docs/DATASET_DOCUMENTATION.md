# FlakyGuard AI: Comprehensive Dataset Documentation

## 1. Provenance & Research Paper Context

- **Research Paper**: *Test Flakiness Across Programming Languages*
- **Authors**: Keila Barbosa, Ronivaldo Ferreira, Gustavo Pinto, Marcelo d'Amorim, Breno Miranda
- **Journal**: *IEEE Transactions on Software Engineering (TSE)*, Vol. 49, No. 4, pp. 2074–2095, April 2023
- **DOI**: [10.1109/TSE.2022.3208864](https://doi.org/10.1109/TSE.2022.3208864)
- **Official Benchmark Repository**: [https://github.com/Test-Flaky/TSE22](https://github.com/Test-Flaky/TSE22)
- **Primary Raw Data Source**: `data/raw/DataBase_GitHub.xlsx` (Sheet: `Data`)

---

## 2. Reconciled Dataset Statistics

The table below reconciles all issue counts, flakiness labels, and unique repository counts across the preprocessing pipeline:

| Dataset Stage | Total Issues | Flaky (T) | False Positive (F) | Not Decided (ND) | Canonical Repositories | Target Task |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Excel Sheet (`Data`)** | **1,543** | 719 | 400 | 424 | 850 | Initial author collection |
| **After Deduplication** | **1,513** | 703 | 400 | 410 | 842 | Removed identical URL/text duplicates |
| **Binary Detection Dataset** | **1,101** | **701** | **400** | *Excluded* | **641** | Task A: Binary Classification |
| **Root-Cause Dataset** | **589** | **589** | *Excluded* | *Excluded* | **349** | Task B: 13-Class Taxonomy |

> [!IMPORTANT]
> **Resolution of the "421 Repositories" Discrepancy**:
> In the raw Excel workbook `DataBase_GitHub.xlsx`, sheet `ProjectsDomain` row 2 reports "421 projects with at least 1 flaky issue". This figure represents the original authors' project filtering count for their domain analysis tab. It is **not** the unique repository count of the binary dataset. The cleaned binary classification dataset ($N=1,101$) contains exactly **641 canonical `{owner}/{repo}` repositories** extracted from `url_issue`.

---

## 3. Programming Language Distribution

The dataset spans five major programming language ecosystems:

| Programming Language | Binary Dataset ($N=1,101$) | Flaky ($Y=1$) | False Positive ($Y=0$) | Flaky Ratio | Root-Cause Dataset ($N=589$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JavaScript (`js`)** | 265 | 179 | 86 | 67.5% | 158 |
| **Python (`python`)** | 222 | 148 | 74 | 66.7% | 118 |
| **Java (`java`)** | 211 | 134 | 77 | 63.5% | 112 |
| **C (`C`)** | 211 | 129 | 82 | 61.1% | 108 |
| **Go (`go`)** | 192 | 111 | 81 | 57.8% | 93 |
| **Total** | **1,101** | **701** | **400** | **63.67%** | **589** |

---

## 4. Root-Cause Taxonomy & Granularity

The root-cause dataset consists of **589 confirmed flaky issues** annotated across 13 fine-grained categories, mapped into 4 macro categories:

| Fine-Grained Category (13 Classes) | Sample Count ($N=589$) | Class % | Macro Category (4 Classes) | Macro Count | Macro % |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Async Wait** | 105 | 17.83% | **Timing & Asynchrony** | 198 | 33.62% |
| **Time** | 46 | 7.81% | Timing & Asynchrony | | |
| **Test Case Timeout** | 42 | 7.13% | Timing & Asynchrony | | |
| **Test Suite Timeout** | 5 | 0.85% | Timing & Asynchrony | | |
| **Concurrency** | 94 | 15.96% | **Concurrency & Non-Determinism** | 144 | 24.45% |
| **Randomness** | 26 | 4.41% | Concurrency & Non-Determinism | | |
| **Unordered Collections** | 24 | 4.07% | Concurrency & Non-Determinism | | |
| **Network** | 86 | 14.60% | **Resource & Environment** | 195 | 33.11% |
| **Platform Dependency** | 57 | 9.68% | Resource & Environment | | |
| **Resource Leak** | 52 | 8.83% | Resource & Environment | | |
| **Test Order Dependency** | 35 | 5.94% | **Test Logic & Assertions** | 52 | 8.83% |
| **Floating Point Operations** | 11 | 1.87% | Test Logic & Assertions | | |
| **Too Restrictive Range** | 6 | 1.02% | Test Logic & Assertions | | |
| **Total** | **589** | **100.0%** | **Total** | **589** | **100.0%** |

---

## 5. Canonical Repository Parsing & Leakage Prevention

### Canonical Regex Resolution:
Many raw rows in `DataBase_GitHub.xlsx` contained an empty or inconsistent `Repository/Project` field (192 rows). Every row, however, contained a valid `URL Issue` link (e.g. `https://github.com/DataDog/dd-trace-py/pull/193`).

The function `parse_canonical_repo` in `src/repository_generalization.py` extracts the standardized `{owner}/{repo}` identifier:
```python
def parse_canonical_repo(url: str, fallback_repo: str = "") -> str:
    match = re.search(r"github\.com/([^/]+)/([^/#?]+)", str(url).strip(), re.IGNORECASE)
    if match:
        owner = match.group(1).lower().strip()
        repo = match.group(2).lower().strip()
        repo = re.sub(r"\.(git|patch|diff)$", "", repo)
        return f"{owner}/{repo}"
    # Fallback to normalized fallback_repo if URL regex fails
    return fallback_repo.strip().lower()
```

### Verification of Zero Leakage:
1. **Target Leakage**: Post-outcome annotations (e.g., `Reviewer 1`, `Reviewer 2`, `Fix Strategy`, `Solution Description`) are strictly excluded from the text feature vector. Only `title`, `body`, and `language` are used.
2. **Train/Test Group Separation**: In all repository-independent experiments, `GroupShuffleSplit` groups samples by `canonical_repo`. For all 10 evaluated random seeds, the intersection between training repositories and test repositories is strictly empty:
   $$\text{Train Repositories} \cap \text{Test Repositories} = \emptyset$$
3. **Vocabulary Leakage Prevention**: In LOLO and repository-independent experiments, the TF-IDF vectorizer vocabulary and IDF weights are fitted **exclusively** on the training partition. The test partition is transformed using the fitted training vectorizer.
