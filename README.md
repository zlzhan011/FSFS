# Fairness-Aware Streaming Feature Selection with Causal Graphs

This repository contains the research code and supplementary material for:

> **Fairness-Aware Streaming Feature Selection with Causal Graphs**  
> Leizhen Zhang, Lusi Li, Di Wu, Sheng Chen, and Yi He  
> *2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)*  
> DOI: [10.1109/SMC54092.2024.11169728](https://doi.org/10.1109/SMC54092.2024.11169728)

The paper introduces **Streaming Feature Selection with Causal Fairness (SFCF)**, an online feature-selection framework that jointly considers predictive accuracy, group fairness, feature-set sparsity, and computational efficiency when features arrive sequentially.

> **Naming note.** The repository is named `FSFS`. The paper uses the method name SFCF. Some experiment scripts and output labels retain the historical names `FS²-RI`, `FS²-AD1`, and `FS²-AD2`; these correspond to `SFCF-RI`, `SFCF-AD1`, and `SFCF-AD2`, respectively.

---

## 1. Motivation

Many real-world machine-learning systems continually receive newly engineered or newly collected features. Examples include features for income prediction, credit-risk assessment, recidivism analysis, fraud detection, advertising, and recommendation. In this **streaming-feature** setting:

1. a newly arriving feature may be irrelevant to the prediction target;
2. it may duplicate information already carried by selected features;
3. it may improve prediction while also acting as a proxy for protected information; or
4. a feature that was previously redundant may become useful after a bias-carrying feature is removed.

Simply deleting the protected attribute is generally insufficient. Non-protected features may still encode or reconstruct sensitive information and can therefore transmit group bias into the resulting model.

SFCF addresses this problem by learning two target-centered causal structures online:

- a **label-centered graph** \(G_Y\), which models feature relevance and redundancy with respect to the prediction label \(Y\); and
- a **protected-attribute-centered graph** \(G_S\), which models feature relevance and redundancy with respect to the protected attribute \(S\).

These structures are updated as each feature arrives. SFCF then removes potentially inadmissible or proxy features and, when appropriate, introduces admissible features to recover predictive information lost during bias mitigation.

---

## 2. Problem Setting

Let the features arrive as a sequence

\[
X = \{X_1, X_2, \ldots, X_D\},
\]

where \(X_i\) is observed at streaming round \(i\). The target label is \(Y\), and \(S\) is a protected attribute such as gender, age, or ethnicity.

At every round, the goal is to select a compact feature subset that:

- preserves predictive accuracy;
- satisfies a group-fairness requirement;
- excludes irrelevant and redundant information when possible; and
- can be updated efficiently without rerunning a fully offline feature-selection procedure over the entire feature space.

The paper evaluates group fairness using **equalized-odds difference (EO)**. Higher accuracy is better, whereas a smaller EO value indicates a smaller disparity between protected and reference groups under this metric.

---

## 3. Method Overview

### 3.1 Incremental causal-graph construction

For each arriving feature, the implementation evaluates its dependence on a target variable through conditional-independence tests. The target is considered twice:

1. \(T = Y\), to construct the label-centered structure; and
2. \(T = S\), to construct the protected-attribute-centered structure.

The framework maintains three feature categories relative to each target:

- **strongly relevant features**, represented through the target's Markov blanket;
- **redundant features**, whose information is covered conditionally by other selected features; and
- **irrelevant features**, which do not provide useful information about the target.

The repository includes both Fisher's \(z\)-test and \(G^2\)/chi-square-based conditional-independence utilities. The low-level online Markov-blanket routine is implemented in:

```text
learning_module/osfs_and_fast_osfs/osfs_z_mb.py
```

A simplified view of the pipeline is:

```text
Incoming feature Xi
        |
        +--> update MB(Y), Redundant(Y), Irrelevant(Y)
        |
        +--> update MB(S), Redundant(S), Irrelevant(S)
                         |
                         v
      identify label-relevant features linked to S
                         |
                         v
       remove or replace potentially inadmissible features
                         |
                         v
          train classifier and evaluate ACC, EO,
          selected-feature ratio, and runtime
```

### 3.2 Inadmissible and admissible features

SFCF treats a non-protected feature as potentially **inadmissible** when it remains causally associated with the protected attribute and may therefore leak sensitive information. The method identifies the overlap between label-relevant features and features represented in the protected-attribute graph.

Removing this overlap can improve fairness but may reduce predictive accuracy. SFCF therefore considers redundant features that can become useful after the inadmissible features have been removed.

### 3.3 SFCF variants

The implementation evaluates three variants:

| Variant | Description |
|---|---|
| **SFCF-RI** | Removes the intersection between the label Markov blanket and the protected-attribute-related feature set. This is the most fairness-oriented variant. |
| **SFCF-AD1** | Starts from SFCF-RI and adds admissible features that are redundant with respect to the label but independent of the protected attribute. Its goal is to recover predictive information without reintroducing protected information. |
| **SFCF-AD2** | Starts from SFCF-RI and adds a broader set of replacement features that are redundant with respect to the protected attribute. This variant may recover more predictive information while accepting a different accuracy–fairness tradeoff. |

In several scripts, these variants appear under the legacy labels:

```text
FS^2-RI   -> SFCF-RI
FS^2-AD1  -> SFCF-AD1
FS^2-AD2  -> SFCF-AD2
```

---

## 4. Repository Structure

```text
FSFS/
├── 2024_SMC_Supplementary.pdf
├── README.md
├── analysis_adult/
├── analysis_communities/
├── analysis_compas/
├── analysis_credit_card/
├── analysis_german/
├── correlation_measure/
│   ├── chi_square_g2_test/
│   └── fisher_z_test/
├── discrimination/
├── learning_module/
│   ├── alpha_investing/
│   ├── group_saola/
│   ├── multi_label/
│   └── osfs_and_fast_osfs/
└── statistical_comparsion/
    └── knnclassify/
```

The directory name `statistical_comparsion/` preserves the spelling used in the original research code.

### Main components

| Path | Purpose |
|---|---|
| `2024_SMC_Supplementary.pdf` | Supplementary material accompanying the SMC 2024 paper. |
| `analysis_adult/` | Data processing, feature construction, SFCF/OSFS experiments, five-fold evaluation, and result export for the Adult dataset. |
| `analysis_communities/` | Preprocessing and fairness-aware streaming-feature experiments for Communities and Crime. |
| `analysis_compas/` | COMPAS acquisition/preprocessing utilities and experimental driver. |
| `analysis_credit_card/` | Credit-card data preprocessing and experimental driver. |
| `analysis_german/` | German Credit preprocessing, classifier utilities, SFCF feature-set construction, and evaluation. |
| `correlation_measure/fisher_z_test/` | Fisher's \(z\)-test and partial-correlation utilities for continuous variables. |
| `correlation_measure/chi_square_g2_test/` | \(G^2\)/chi-square conditional-independence utilities for discrete variables. |
| `learning_module/osfs_and_fast_osfs/` | Core online feature-selection and Markov-blanket routines, including relevance and redundancy analysis. |
| `learning_module/multi_label/` | Logistic-regression, MLP, Gaussian-NB, XGBoost, and Keras helper routines used by experiments. |
| `discrimination/` | A custom group-discrimination-score implementation retained from the experimental code. |
| `statistical_comparsion/knnclassify/` | K-nearest-neighbor evaluation utilities retained from the original codebase. |

---

## 5. Benchmark Datasets and Experiment Drivers

The paper evaluates SFCF on five datasets commonly used in feature-selection or algorithmic-fairness research.

| Dataset | Primary repository location | Representative experiment driver |
|---|---|---|
| Adult | `analysis_adult/` | `analysis_adult/osfs_test_adult_odds_KF.py` |
| Communities and Crime | `analysis_communities/` | `analysis_communities/osfs_communities_odds_KF.py` |
| COMPAS | `analysis_compas/` | `analysis_compas/osfs_compas_odds_KF.py` |
| Credit Card | `analysis_credit_card/` | `analysis_credit_card/osfs_credit_card_odds_KF.py` |
| German Credit | `analysis_german/` | `analysis_german/osfs_german_odds.py` |

The dataset-specific scripts define preprocessing, label construction, protected attributes, feature ordering, and evaluation logic. Consult both the paper and `2024_SMC_Supplementary.pdf` before changing these choices.

### Data availability note

The current repository snapshot primarily contains source code and supplementary material; it does **not** provide a standardized, self-contained download of all five raw datasets. Obtain each dataset from its original provider, comply with its license and terms of use, and then update the local paths in the corresponding experiment script.

---

## 6. Installation

### 6.1 Clone the repository

```bash
git clone https://github.com/zlzhan011/FSFS.git
cd FSFS
```

### 6.2 Create an environment

The original code was developed as research scripts and does not include a pinned environment or lockfile. Python 3.9 is a practical compatibility starting point.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 6.3 Install dependencies

Core numerical, learning, fairness, and spreadsheet dependencies:

```bash
pip install numpy pandas scipy scikit-learn fairlearn openpyxl
```

Additional classifiers and visualization utilities used by parts of the repository:

```bash
pip install xgboost lightgbm matplotlib seaborn plotly
```

Keras/TensorFlow helpers are required only for scripts that call the Keras MLP implementation:

```bash
pip install tensorflow keras
```

Because dependency versions are not pinned, a modern environment may require small compatibility edits. In particular, recent NumPy versions removed `np.float`; replace it with `float` or `np.float64` if that error occurs.

---

## 7. Configuration Before Running

The current code preserves paths and conventions from the original experimental environment. Before execution:

1. open the relevant dataset driver;
2. replace hard-coded Windows or Linux paths such as `E:\...` or `/code/...`;
3. set the dataset input paths;
4. select an output directory for spreadsheets and models;
5. verify the label and protected-attribute indices;
6. confirm whether the data are treated as continuous or discrete;
7. verify the conditional-independence significance level, typically `alpha=0.01` in the included experiments; and
8. place the repository root on `PYTHONPATH`.

On Linux or macOS:

```bash
export PYTHONPATH="$PWD:$PWD/analysis_adult:$PYTHONPATH"
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\analysis_adult;$env:PYTHONPATH"
```

Some scripts use dataset-local imports rather than fully qualified package imports. If an import error occurs, add that dataset directory to `PYTHONPATH` or convert the import to an absolute repository import.

---

## 8. Running the Experiments

After configuring dataset and output paths, run a dataset-specific driver from the repository root. For example:

```bash
python analysis_adult/osfs_test_adult_odds_KF.py
```

Other representative drivers are:

```bash
python analysis_communities/osfs_communities_odds_KF.py
python analysis_compas/osfs_compas_odds_KF.py
python analysis_credit_card/osfs_credit_card_odds_KF.py
python analysis_german/osfs_german_odds.py
```

These scripts are research artifacts rather than a unified command-line package. They may require dataset-specific path changes, import adjustments, or compatibility updates before running in a new environment.

### 8.1 Low-level Markov-blanket routine

The core online feature-selection routine can be invoked directly:

```python
from learning_module.osfs_and_fast_osfs.osfs_z_mb import osfs_z_mb

selected_features, redundant_features, irrelevant_features, elapsed = osfs_z_mb(
    data1=data,
    class_index=data.shape[1] - 1,
    alpha=0.01,
    max_k=100,
)
```

Here, `data` is a two-dimensional NumPy array and the target variable is stored at `class_index`. This routine learns feature relevance and redundancy relative to **one** target. The dataset drivers invoke analogous analysis for both \(Y\) and \(S\), derive the SFCF feature sets, train classifiers, and compute fairness and accuracy.

### 8.2 Reproduction workflow

A typical end-to-end experiment follows this sequence:

1. load and preprocess a benchmark dataset;
2. create the prescribed feature stream;
3. split the data into five folds/runs;
4. construct the label-centered Markov blanket;
5. treat the protected attribute as a second target and construct its Markov blanket;
6. identify the intersection of label-relevant and protected-related features;
7. construct the SFCF-RI, SFCF-AD1, and SFCF-AD2 feature sets;
8. train the downstream classifier;
9. evaluate accuracy and equalized-odds difference;
10. record the number or proportion of selected features and runtime; and
11. export results to spreadsheet files.

To reproduce the principal paper comparison, use the logistic-regression setting and the evaluation protocol described in the paper and supplementary material. Other classifiers in the repository are exploratory utilities and may not correspond exactly to every result in the main paper.

---

## 9. Evaluation Metrics

The experiments report four principal dimensions:

| Metric | Interpretation |
|---|---|
| **Accuracy (ACC)** | Predictive correctness; larger is better. |
| **Equalized-odds difference (EO)** | Group disparity under equalized odds; smaller is better. |
| **Selected-feature ratio / count** | Sparsity of the selected subset; smaller is generally better when predictive and fairness performance are retained. |
| **Runtime** | Cost of processing the streaming features and constructing the selected subset. |

The custom function in `discrimination/calculate_discrimination.py` computes a difference in positive-outcome rates between two groups. The main paper, however, reports **equalized odds**, and the dataset drivers also call `fairlearn.metrics.equalized_odds_difference`. Do not treat the custom discrimination score and EO as interchangeable.

---

## 10. Paper-Reported Results

The following values summarize the aggregate results reported in the paper relative to the all-feature baseline:

| Variant | Average EO reduction | Average relative accuracy decrease | Average selected-feature ratio |
|---|---:|---:|---:|
| **SFCF-RI** | 39.82% | 5.80% | 14.17% |
| **SFCF-AD1** | 53.45% | 5.55% | 25.56% |
| **SFCF-AD2** | 47.16% | 4.92% | 45.97% |

Additional paper-level observations include:

- SFCF-AD1 achieved an average EO value of **0.103** while maintaining an average accuracy of **0.748**.
- Compared with OSFS, the three SFCF variants reduced EO while keeping the average accuracy change relatively small.
- The proposed variants had an average reported runtime of approximately **2.394 seconds** under the paper's experimental setup.
- The experiments show that removing only the protected attribute is not sufficient when other features can carry protected information.
- Adding admissible replacement features can recover predictive information lost when inadmissible features are removed.

These are aggregated results from the paper, not universal guarantees. Exact performance depends on the dataset, feature order, preprocessing, protected attribute, classifier, conditional-independence test, significance threshold, software environment, and hardware.

---

## 11. Reproducibility Notes

Please account for the following before comparing new results with the paper:

- **Feature order matters.** This is a streaming-feature setting, so changing the arrival order can change the selected subset.
- **Raw datasets are not normalized into one repository-wide format.** Each dataset driver has its own preprocessing assumptions.
- **Paths are environment-specific.** Several scripts contain absolute local paths and must be edited.
- **Dependency versions are unpinned.** Numerical or model-library changes may affect results.
- **Legacy naming remains in code.** `FS^2-*` output labels map to the `SFCF-*` names used in the paper.
- **Compiled cache files may be ignored.** Source `.py` files, not `__pycache__` or `.pyc` files, should be used.
- **Not every paper baseline is exposed as one uniform runnable command.** Reproducing the complete comparison may require the original baseline implementations and the detailed supplementary protocol.
- **Randomness must be controlled.** Preserve the seeds, fold construction, and feature order used by the corresponding driver.
- **Observational causal structure should be interpreted carefully.** The learned graphs operationalize conditional-independence relationships used by SFCF; they do not by themselves establish every real-world causal relationship.

---

## 12. Responsible Use

This repository is a research artifact for studying fairness-aware online feature selection. A low equalized-odds difference on a benchmark does not establish that a deployed system is fair, lawful, safe, or appropriate for a particular decision context.

Before applying the method to consequential decisions:

- identify the legally and contextually relevant protected groups;
- justify the selected fairness definition;
- examine subgroup sample sizes and data quality;
- assess distribution shift;
- evaluate multiple fairness and utility metrics;
- document feature provenance and proxy risks; and
- include domain experts and affected stakeholders in system review.

---

## 13. Citation

Please cite the paper when using this repository:

```bibtex
@inproceedings{zhang2024fairness,
  author    = {Leizhen Zhang and Lusi Li and Di Wu and Sheng Chen and Yi He},
  title     = {Fairness-Aware Streaming Feature Selection with Causal Graphs},
  booktitle = {2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages     = {858--865},
  year      = {2024},
  doi       = {10.1109/SMC54092.2024.11169728}
}
```

---

## 14. Acknowledgment

The paper acknowledges support in part from the U.S. National Science Foundation under Grants **CNS-2245918**, **IIS-2245946**, and **IIS-2236578**, and from the **Commonwealth Cyber Initiative (CCI)**.

---

## 15. License

This repository currently does not include an explicit software license. Public availability alone does not define redistribution or reuse rights. Please contact the authors before redistributing or incorporating the code into another project, or add an appropriate license after obtaining agreement from the relevant contributors and institutions.
