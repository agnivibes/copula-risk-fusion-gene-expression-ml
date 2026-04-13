# Copula-Based Fusion of Clinical and Gene-Expression Machine Learning Risk Scores for Breast Cancer Risk Stratification 🧬📊

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%2Fxgboost-orange)](https://scikit-learn.org/stable/)
[![Lifelines](https://img.shields.io/badge/Survival-Lifelines-FF6F61)](https://lifelines.readthedocs.io/en/latest/)
[![Gene Expression](https://img.shields.io/badge/Bio-Gene%20Expression-2E8B57)](https://en.wikipedia.org/wiki/Gene_expression_profiling)
[![Copulas](https://img.shields.io/badge/Stats-Copula%20Modeling-6E40AA)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
[![METABRIC](https://img.shields.io/badge/Data-METABRIC-blue)](https://www.cbioportal.org/study/summary?id=brca_metabric)
[![TCGA](https://img.shields.io/badge/Data-TCGA-green)](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018)

This repository contains the complete, reproducible machine learning pipeline for our study on integrating multi-view biomedical data (clinical and gene-expression) using copula theory.

We introduce a fusion framework that models the dependency structure between disparate risk scores using Gaussian, Clayton, Gumbel, and Frank copulas. We evaluate the nature of joint risk dependence in breast cancer patients using the METABRIC cohort (primary analysis) and validate the framework in an independent TCGA cohort (external validation).

All results, methodological details, and discussions are provided in our accompanying research paper. This repository focuses strictly on code and reproducibility.

---

## 📂 Dataset Information

### Primary Dataset: METABRIC

The primary analysis relies on the processed METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset accessed via Kaggle.

**Source:** [Breast Cancer Gene Expression Profiles (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)

This is a publicly available, pre-processed CSV file (`METABRIC_RNA_Mutation.csv`) that combines curated clinical attributes, mRNA expression z-scores for 489 genes, binary mutation indicators, and survival outcomes for 1,904 patients. The file was derived from the original METABRIC studies:

- Curtis et al., *Nature*, 2012
- Pereira et al., *Nature Communications*, 2016

Our analysis uses this processed file directly as provided on Kaggle. No additional preprocessing was applied to the source data prior to the steps described in our code and manuscript.

### External Validation Dataset: TCGA

For external validation, we constructed an independent breast cancer dataset from The Cancer Genome Atlas (TCGA).

**Source:** [Breast Invasive Carcinoma (TCGA, PanCancer Atlas)](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018)

The TCGA dataset was built by downloading clinical patient data and mRNA expression z-scores from cBioPortal, then cleaning, harmonizing, and merging them using the R preprocessing script included in this repository (`R-code-for-tcga-data-generation.txt`). The harmonized dataset retains 476 gene-expression features shared between METABRIC and TCGA, along with shared clinical predictors (age at diagnosis and tumour stage).

---

## 📦 Requirements

Python 3.11+ and R 4.0+ (for TCGA data construction only).

Install required Python packages via:

```bash
pip install numpy pandas scipy scikit-learn xgboost lifelines matplotlib
```

For TCGA data construction, the R script requires `data.table` and `dplyr`.

---

## 🚀 Getting Started

```bash
git clone https://github.com/agnivibes/copula-risk-fusion-gene-expression-ml.git
cd copula-risk-fusion-gene-expression-ml
```

### Running the Analysis

This repository contains two main analysis scripts:

**Script 1 — Primary METABRIC Analysis** (`copula-risk-fusion-gene-expression-ml.py`):

Runs the full primary analysis pipeline on the METABRIC cohort, including clinical and gene-expression ML model training, copula fitting (Gaussian, Clayton, Gumbel, Frank), goodness-of-fit testing, joint risk stratification, Kaplan–Meier survival analysis, and competing-risks cumulative incidence analysis. Outputs are saved to `study1_outputs/`.

```bash
python copula-risk-fusion-genomics-ml.py
```

**Script 2 — External Validation in TCGA** (`external_validation_tcga.py`):

Runs the harmonized external-validation pipeline. Trains clinical and gene-expression models on harmonized METABRIC data, fits copulas on the training scores, and evaluates all three scores (clinical, gene-expression, copula-fused) in the independent TCGA cohort. Outputs are saved to `harmonized_external_validation_outputs/`.

```bash
python external_validation_tcga.py
```

Both scripts require `METABRIC_RNA_Mutation.csv` in the working directory. Script 2 additionally requires `tcga_final_dataset.csv`. Download the METABRIC CSV from Kaggle; to construct the TCGA dataset, run the provided R script (`R-code-for-tcga-data-generation.txt`) after downloading the raw clinical and expression files from cBioPortal.

---

## 🔬 Research Paper

Aich, A., Hewage, S., Murshed, M. (2025). Copula Based Fusion of Clinical and Gene Expression Machine Learning Risk Scores for Breast Cancer Risk Stratification. [Manuscript under review]

## 📊 Citation

If you use this code or method in your own work, please cite:

```bibtex
@article{Aich2025CopulaFusionGeneExprML,
  title   = {Copula Based Fusion of Clinical and Gene Expression Machine Learning Risk Scores for Breast Cancer Risk Stratification},
  author  = {Aich, Agnideep and Hewage, Sameera and Murshed, Md Monzur},
  journal = {},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## 📬 Contact

For questions or collaborations, feel free to contact:

**Agnideep Aich**
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).
