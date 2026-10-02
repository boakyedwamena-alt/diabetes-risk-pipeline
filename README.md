# Early Diabetes Risk Detection: Machine Learning Pipeline

An end-to-end machine learning pipeline for early diabetes risk prediction on the Pima Indians Diabetes dataset, built as part of an MSc Data Analytics dissertation. It follows [TRIPOD](https://www.tripod-statement.org/) reporting guidance and combines a biomedical science background with applied machine learning, with the aim of producing predictions that are both accurate and clinically interpretable.

## Summary

- A stacking ensemble reached **recall 0.76** on a held-out test set, catching 41 of 54 patients with diabetes, compared with 38 for a simple logistic regression baseline.
- Overall discrimination was similar across models (ROC AUC about 0.81). With only 154 test patients, the ensemble's advantage over the baseline is **not statistically clear** (see [Results](#results)).
- SHAP analysis shows glucose, BMI and insulin drive the predictions, which is consistent with clinical expectations.

## Overview

The pipeline covers the full analytics lifecycle:

1. **EDA and data quality checks:** identifying biologically implausible zero values (for example zero blood pressure or BMI) and structural issues in the raw data.
2. **Preprocessing:** KNN imputation for missing values and medically informed IQR-based outlier capping, fitted on the training set only to prevent data leakage.
3. **Feature engineering:** age, BMI and insulin binning, a glucose-BMI interaction term, and a high-pregnancies flag.
4. **Feature selection:** mutual information scoring combined with Variance Inflation Factor (VIF) checks to remove redundant, collinear features.
5. **Modelling:** five base classifiers (Logistic Regression, Random Forest, XGBoost, SVM, Gaussian Naive Bayes), each tuned with grid search and evaluated with and without ADASYN oversampling to address class imbalance.
6. **Stacking ensemble:** a tuned XGBoost meta-model combining the five base learners, selected via randomized search.
7. **Evaluation:** precision, recall, F1, ROC AUC, PR AUC, balanced accuracy and MCC on a held-out test set, plus overfitting diagnostics and training/inference timing.
8. **Interpretability:** SHAP values and partial dependence plots for the stacking ensemble and its strongest base learners.

## Results

Evaluated once on a held-out, stratified test set (n = 154, of whom 54 have diabetes). The models were trained on 614 patients, with random seed 42 and a default decision threshold of 0.5.

| Model | Recall | Precision | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Baseline: logistic regression, raw features | 0.704 | 0.603 | 0.650 | 0.813 | 0.673 |
| XGBoost (ADASYN) | 0.722 | 0.591 | 0.650 | 0.821 | 0.644 |
| **Stacking ensemble (final model)** | **0.759** | 0.612 | 0.678 | 0.809 | 0.655 |

The baseline uses the same split but none of the pipeline's steps (no KNN imputation, outlier capping, feature engineering, feature selection or ADASYN), only median imputation of the invalid zeros and scaling. The full results for all 12 models are in [`results/test_set_metrics.csv`](results/test_set_metrics.csv), and confusion matrices are in [`results/test_set_confusion_matrices.csv`](results/test_set_confusion_matrices.csv).

![ROC and precision-recall curves on the test set](figures/roc_pr_comparison.png)

### How to read these results

- The stacking ensemble had the highest recall and the fewest missed cases (13 false negatives, against 16 for the baseline). This matters because missing a true case is the costlier error in screening.
- Discrimination was essentially the same across models, and the baseline's PR AUC was slightly higher than the ensemble's.
- The test set is small, so differences are within sampling uncertainty. For recall, the ensemble minus baseline difference was +5.6 points, with a 95% bootstrap interval of -5.0 to +17.2 points ([`results/bootstrap_ci.csv`](results/bootstrap_ci.csv)). The data support "the ensemble is competitive and catches slightly more cases", not "the ensemble is clearly better".
- The engineered pipeline did not clearly beat a simple logistic regression on this dataset. That is a common finding on small tabular datasets and is reported here as it is.

### Interpretability

![SHAP summary for XGBoost](figures/shap_summary_xgboost.png)

Glucose, BMI and insulin contribute most to the predictions, with age, blood pressure, pregnancies and skin thickness contributing less. Higher glucose and BMI push predictions towards higher diabetes risk. Partial dependence plots are in [`figures/Partial_Dependence_Plots.png`](figures/Partial_Dependence_Plots.png), and the contribution of each base model to the stacking ensemble is in [`figures/Contribution_to_Stacking_Ensemble.png`](figures/Contribution_to_Stacking_Ensemble.png).

## Limitations

- **Small dataset:** 768 patients, with only 154 in the test set, so all metrics carry wide uncertainty (for example, the 95% interval for baseline recall is roughly 0.58 to 0.83).
- **Single population:** the data come from Pima Indian women aged 21 and over. Results may not generalise to men, other age groups or other populations, and the model has not been externally validated.
- **Missing-data assumption:** zeros in glucose, blood pressure, skin thickness, insulin and BMI were treated as missing values and imputed. This is an assumption about how the data were recorded.
- **Choice of illustrated models:** hyperparameters and resampling choices were selected by cross-validation on the training set only. The base models shown in the figures and SHAP plots were chosen for illustration after viewing test-set results.
- **Fixed threshold:** results use the default 0.5 decision threshold. A screening tool would need a threshold chosen for its clinical setting.
- **Not a clinical tool:** this is a research and portfolio project. It is not validated for clinical use and must not be used to make medical decisions.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `XGBoost` · `SHAP` · `statsmodels` · `seaborn` / `matplotlib`

## Project Structure

```
.
├── diabetes_pipeline.py     # Full pipeline: EDA → preprocessing → modelling → evaluation → SHAP
├── bootstrap_ci.py          # 95% bootstrap intervals for baseline vs stacking ensemble
├── requirements.txt         # Python dependencies
├── data/                    # Place diabetes.csv here (not included, see below)
├── figures/                 # Key plots used in this README
├── results/                 # Test-set metrics, confusion matrices, bootstrap intervals, run info
├── saved_models/            # Trained models (generated on run, git-ignored)
└── saved_evaluations/       # Predictions and evaluation objects (generated on run, git-ignored)
```

## Dataset

This project uses the Pima Indians Diabetes dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases (Smith et al., 1988, *Proceedings of the Symposium on Computer Applications and Medical Care*, 261-265). It was previously hosted on Kaggle at `uciml/pima-indians-diabetes-database`, but that listing is no longer available. The dataset can still be downloaded from this mirror: [github.com/npradaschnor/Pima-Indians-Diabetes-Dataset](https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset). Download `diabetes.csv` and place it in a `data/` folder at the project root.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/boakyedwamena-alt/diabetes-risk-pipeline.git
cd diabetes-risk-pipeline

# Install dependencies
pip install -r requirements.txt

# Add the dataset
mkdir -p data
# place diabetes.csv inside data/

# Run the pipeline (takes roughly 10 minutes on a single CPU core)
python diabetes_pipeline.py

# Optional: bootstrap confidence intervals (run after the pipeline)
python bootstrap_ci.py
```

To run without plot windows opening (for example on a server), use `MPLBACKEND=Agg python diabetes_pipeline.py`.

**Reproducibility:** all random seeds are fixed (42), and two independent runs produced identical metrics. Results can differ slightly with other package versions, so the versions used are recorded in [`results/run_info.json`](results/run_info.json).

## Methodology Notes

- All preprocessing (imputation, scaling, outlier capping, feature engineering) is **fit on the training set only** and applied to the test set afterwards, to avoid data leakage. ADASYN is applied inside the cross-validation pipeline, so only training folds are oversampled.
- Class imbalance is addressed by comparing each model with and without **ADASYN** oversampling and keeping whichever performs better on cross-validated PR AUC.
- Model selection prioritises **recall** and **PR AUC** over raw accuracy, reflecting the clinical cost of missing a true diabetes case (false negatives) versus a false alarm.
- The pipeline follows TRIPOD reporting guidance. A completed TRIPOD checklist is not included.

## Author

**Emmanuel Dwamena**, MSc Data Analytics (Distinction), University of Portsmouth
[LinkedIn](https://linkedin.com/in/emmanuel-dwamena)
