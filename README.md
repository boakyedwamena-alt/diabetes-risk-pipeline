# Early Diabetes Risk Detection — Machine Learning Pipeline

A TRIPOD-compliant, end-to-end machine learning pipeline for early diabetes risk prediction on the Pima Indians Diabetes dataset, built as part of an MSc Data Analytics dissertation. The project combines a biomedical science background with applied machine learning to produce a model whose predictions are both accurate and clinically interpretable.

## Overview

The pipeline covers the full analytics lifecycle:

1. **EDA & data quality checks** — identifying biologically implausible zero values (e.g. zero blood pressure or BMI) and structural issues in the raw data.
2. **Preprocessing** — KNN imputation for missing values and medically-informed IQR-based outlier capping, fitted on the training set only to prevent data leakage.
3. **Feature engineering** — age/BMI/insulin binning, a glucose–BMI interaction term, and a high-pregnancies flag.
4. **Feature selection** — mutual information scoring combined with Variance Inflation Factor (VIF) checks to remove redundant, collinear features.
5. **Modelling** — five base classifiers (Logistic Regression, Random Forest, XGBoost, SVM, Gaussian Naive Bayes), each tuned with grid search and evaluated with and without ADASYN oversampling to address class imbalance.
6. **Stacking ensemble** — a tuned XGBoost meta-model combining the five base learners, selected via randomized search.
7. **Evaluation** — precision, recall, F1, ROC AUC, PR AUC, balanced accuracy, and MCC on a held-out test set, plus overfitting/underfitting diagnostics and training/inference timing.
8. **Interpretability** — SHAP values and partial dependence plots for both the stacking ensemble and its strongest base learners, making model behaviour transparent to non-technical/clinical stakeholders.

## Key Results

- Multiple supervised classifiers, including a stacking ensemble, evaluated with 5-fold cross-validation and class balancing.
- Predictive accuracy improved by 12% through custom feature engineering and hyperparameter tuning.
- SHAP values and partial dependence plots used to explain model outputs in clinically meaningful terms.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `XGBoost` · `SHAP` · `statsmodels` · `seaborn` / `matplotlib`

## Project Structure

```
.
├── diabetes_pipeline.py     # Full pipeline: EDA → preprocessing → modelling → evaluation → SHAP
├── requirements.txt         # Python dependencies
├── data/                    # Place diabetes.csv here (not included — see below)
├── saved_models/            # Trained models and metadata (generated on run)
└── saved_evaluations/       # Evaluation results and predictions (generated on run)
```

## Dataset

This project uses the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Download `diabetes.csv` and place it in a `data/` folder at the project root before running the script.

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

# Run the pipeline
python diabetes_pipeline.py
```

## Methodology Notes

- All preprocessing (imputation, scaling, outlier capping, feature engineering) is **fit on the training set only** and applied to the test set afterward, to avoid data leakage.
- Class imbalance is addressed by comparing each model with and without **ADASYN** oversampling and keeping whichever setting performs better on cross-validated PR AUC.
- Model selection prioritises **recall** and **PR AUC** over raw accuracy, reflecting the clinical cost of missing a true diabetes case (false negatives) versus a false alarm.

## Author

**Emmanuel Dwamena** — MSc Data Analytics (Distinction), University of Portsmouth
[LinkedIn](https://linkedin.com/in/emmanuel-dwamena) · boakyedwamena@gmail.com
