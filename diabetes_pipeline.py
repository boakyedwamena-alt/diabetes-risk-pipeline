"""
Early Diabetes Risk Detection — End-to-End ML Pipeline
========================================================
TRIPOD-compliant pipeline for early diabetes risk prediction on the
Pima Indians Diabetes dataset: EDA, medically-informed preprocessing,
feature engineering, model selection (with ADASYN resampling), a
stacking ensemble, evaluation, and SHAP / PDP interpretability.

Author: Emmanuel Dwamena
"""

import os
import json
import time
import random
import shutil
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter

import joblib
import shap
from IPython.display import display

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict,
    cross_val_score, learning_curve, RepeatedStratifiedKFold, RandomizedSearchCV,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    make_scorer, accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,
    precision_recall_curve, average_precision_score, roc_curve, auc,
)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline, Pipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# =====================================================================
# 1. Dataset Loading and Inspection
# =====================================================================
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
file_name = "diabetes.csv"
file_path = os.path.join(base_dir, file_name)

print("\nLoading the Pima Indians Diabetes dataset...")
pid = pd.read_csv(file_path)
print("Dataset loaded successfully. Preview of data:")
print(pid.head())


# =====================================================================
# 2. Dataset Understanding
# =====================================================================
pid.info()
print("\nSummary statistics of the dataset:")
print(pid.describe())

pid.hist(bins=30, figsize=(15, 10))
plt.suptitle('Variable Distributions')
plt.show()

num_duplicates = pid.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")


# =====================================================================
# 3. Dataset Preparation
# =====================================================================
print("Starting data split...")
X = pid.drop('Outcome', axis=1)
y = pid['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

train_df = X_train.copy()
test_df = X_test.copy()


# =====================================================================
# 4. Exploratory Data Analysis (EDA) — invalid zero handling
# =====================================================================
print("\nIdentifying columns with invalid zero values...")
zeros_cols = [col for col in train_df.columns if (train_df[col] == 0).sum() > 0]
print("Columns with zero values and their counts (in TRAIN set):")
for col in zeros_cols:
    zero_count = (train_df[col] == 0).sum()
    print(f"{col}: {zero_count} zeros")

zero_as_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nReplacing 0s with NaNs in TRAIN set (for imputation)...")
for col in zero_as_nan:
    train_df[col] = train_df[col].replace(0, np.nan)
    nan_count = train_df[col].isna().sum()
    print(f"{col}: NaNs = {nan_count}")

test_df[zero_as_nan] = test_df[zero_as_nan].replace(0, np.nan)
print("\nZeros replacement with NaNs completed on TEST set.")

joblib.dump(zero_as_nan, 'zero_as_nan.pkl')

print("Dataset loaded successfully. Preview of data:")
print(train_df.head(10))


# =====================================================================
# 5. KNN Imputation (fit on train only — no leakage)
# =====================================================================
print("\nImputing missing values in TRAIN set using KNN imputer...")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[zero_as_nan])
test_scaled = scaler.transform(test_df[zero_as_nan])

imputer = KNNImputer(n_neighbors=5)
train_imputed = imputer.fit_transform(train_scaled)
print("Imputation complete on TRAIN set.")

print("Imputing missing values in TEST set using trained KNN imputer...")
test_imputed = imputer.transform(test_scaled)
print("Imputation complete on TEST set. No data leakage introduced.")

train_df[zero_as_nan] = scaler.inverse_transform(train_imputed)
test_df[zero_as_nan] = scaler.inverse_transform(test_imputed)
print("Standardized values reverted to original scale in both TRAIN and TEST sets.")

assert train_df[zero_as_nan].isna().sum().sum() == 0, "NaNs still present in train_df after imputation!!"
assert test_df[zero_as_nan].isna().sum().sum() == 0, "NaNs still present in test_df after imputation!!"

joblib.dump(scaler, 'standard_scaler.pkl')
joblib.dump(imputer, 'knn_imputer.pkl')

print("Dataset loaded successfully. Preview of data:")
print(train_df.head(10))

print("\nGenerating descriptive statistics for the train dataset...")
print(train_df.describe())


# =====================================================================
# 6. Outlier Inspection and Medically-Informed Capping
# =====================================================================
selected_cols = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
num_cols = 4
num_rows = (len(selected_cols) + num_cols - 1) // num_cols
plt.figure(figsize=(num_cols * 5, num_rows * 4))
for i, col in enumerate(selected_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=train_df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig("Boxplot_before.png")
plt.show()

outlier_caps = {}

def calculate_outlier_thresholds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def apply_outlier_caps(df, column, lower, upper, direction='both'):
    if direction == 'both':
        df[column] = np.where(df[column] > upper, upper,
                              np.where(df[column] < lower, lower, df[column]))
    elif direction == 'upper':
        df[column] = np.where(df[column] > upper, upper, df[column])
    elif direction == 'lower':
        df[column] = np.where(df[column] < lower, lower, df[column])

for col, direction in [('SkinThickness', 'upper'),
                       ('Insulin', 'upper'),
                       ('BloodPressure', 'lower'),
                       ('BMI', 'upper')]:
    lower, upper = calculate_outlier_thresholds(train_df, col)
    outlier_caps[col] = {'lower': lower, 'upper': upper, 'direction': direction}
    apply_outlier_caps(train_df, col, lower, upper, direction)

for col, params in outlier_caps.items():
    apply_outlier_caps(test_df, col, params['lower'], params['upper'], params['direction'])

print("\nOutlier treatment applied safely without data leakage.")
joblib.dump(outlier_caps, 'outlier_caps.pkl')

plt.figure(figsize=(num_cols * 5, num_rows * 4))
for i, col in enumerate(selected_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=train_df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig("Boxplot_after.png")
plt.show()


# =====================================================================
# 7. Feature Engineering
# =====================================================================
print("\nPerforming feature engineering on TRAIN data...")
age_bins = [20, 30, 40, 50, 60, 85]
bmi_bins = [0, 18.5, 24.9, 29.9, 100]
insulin_bins = [-np.inf, 16, 166, train_df['Insulin'].max() + 1]

age_labels = ['20–29', '30–39', '40–49', '50–59', '60+']
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
insulin_labels = ['Low', 'Normal', 'High']

train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels, right=False)
train_df['BMI_Category'] = pd.cut(train_df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
train_df['InsulinStatus'] = pd.cut(train_df['Insulin'], bins=insulin_bins, labels=insulin_labels, right=False)
train_df['Glucose_BMI_Interaction'] = train_df['Glucose'] * train_df['BMI']
train_df['HighPregnancies'] = (train_df['Pregnancies'] > 5).astype(int)

print("Feature engineering done. Sample:")
print(train_df.head())

train_df = pd.get_dummies(train_df, columns=['BMI_Category', 'InsulinStatus'], drop_first=True)
print("One-hot encoding done. Columns now:", train_df.columns.tolist())

feature_engineering_config = {
    'age_bins': age_bins,
    'bmi_bins': bmi_bins,
    'insulin_bins': insulin_bins,
    'age_labels': age_labels,
    'bmi_labels': bmi_labels,
    'insulin_labels': insulin_labels,
    'train_columns': train_df.columns.tolist()
}
joblib.dump(feature_engineering_config, 'feature_engineering_config.pkl')

print("\nPerforming feature engineering on TEST data using saved bins (no data leakage introduced)...")
config = joblib.load('feature_engineering_config.pkl')
test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=config['age_bins'], labels=config['age_labels'], right=False)
test_df['BMI_Category'] = pd.cut(test_df['BMI'], bins=config['bmi_bins'], labels=config['bmi_labels'], right=False)
test_df['InsulinStatus'] = pd.cut(test_df['Insulin'], bins=config['insulin_bins'], labels=config['insulin_labels'], right=False)
test_df['Glucose_BMI_Interaction'] = test_df['Glucose'] * test_df['BMI']
test_df['HighPregnancies'] = (test_df['Pregnancies'] > 5).astype(int)

test_df = pd.get_dummies(test_df, columns=['BMI_Category', 'InsulinStatus'], drop_first=True)
print("One-hot encoding done. Columns now:", test_df.columns.tolist())

test_df = test_df.reindex(columns=config['train_columns'], fill_value=0)


# =====================================================================
# 8. Type Cleanup
# =====================================================================
print("\nSummary of train_df (after feature engineering):")
train_df.info()

train_df['AgeGroup'] = train_df['AgeGroup'].cat.codes
test_df['AgeGroup'] = test_df['AgeGroup'].cat.codes

bool_cols = train_df.select_dtypes(include='bool').columns
train_df[bool_cols] = train_df[bool_cols].astype(int)
test_df[bool_cols] = test_df[bool_cols].astype(int)

assert all(train_df[bool_cols].dtypes == test_df[bool_cols].dtypes), "Mismatch in dtypes between train and test"
print("Assert passed: Bool column dtypes match in train and test.")

joblib.dump(bool_cols.tolist(), 'bool_columns.pkl')
train_df.info()


# =====================================================================
# 9. Scaling
# =====================================================================
print("\nScaling continuous features using RobustScaler...")
scaler = RobustScaler()
continuous_features = ['Glucose_BMI_Interaction', 'Glucose', 'BMI', 'Insulin', 'BloodPressure']

X_train_scaled = train_df.copy()
X_test_scaled = test_df.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train_scaled[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test_scaled[continuous_features])

print("Scaling done for Train data. Sample after scaling:")
print(X_train_scaled[continuous_features].head())
print("=" * 50)
print("Scaling done for TEST data. Sample after scaling:")
print(X_test_scaled[continuous_features].head())

joblib.dump(scaler, 'robust_scaler.pkl')

X_train_scaled['Outcome'] = y_train.values
print(X_train_scaled.columns.tolist())


# =====================================================================
# 10. Feature Selection — Mutual Information + VIF
# =====================================================================
print("\nCalculating Mutual Information for feature selection...")
X_fs = X_train_scaled.drop(columns='Outcome')
y_fs = X_train_scaled['Outcome']

mi_scores = mutual_info_classif(X_fs, y_fs, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': X_fs.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=mi_df, x='MI_Score', y='Feature', palette='viridis')
plt.title("Mutual Information Scores")
plt.tight_layout()
plt.savefig("mutual_information_scores.png")
plt.show()

threshold = 0.02
top_features = mi_df[mi_df['MI_Score'] > threshold]['Feature'].tolist()
print(f"\nSelected {len(top_features)} features with MI > {threshold}:")
print(top_features)
joblib.dump(top_features, 'top_features.pkl')

X_train_selected = X_fs[top_features]
print("Selected features for the TRAIN data. Columns now:")
print(X_train_selected.columns.tolist())
print("=" * 50)

X_test_selected = X_test_scaled[top_features]
print("Selected features for the TEST data. Columns now:")
print(X_test_selected.columns.tolist())

print("\nChecking Variance Inflation Factor (VIF) to detect multicollinearity...")
X_vif = X_train_selected.select_dtypes(include=[np.number]).copy()

constant_columns = [col for col in X_vif.columns if X_vif[col].nunique() <= 1]
if constant_columns:
    print(f"Dropping constant columns (zero variance): {constant_columns}")
    X_vif.drop(columns=constant_columns, inplace=True)

vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.sort_values(by='VIF', ascending=False))

high_vif_features = ['Glucose_BMI_Interaction', 'Age', 'InsulinStatus_Normal', 'InsulinStatus_High', 'BMI_Category_Obese']

X_train_final = X_train_selected.drop(columns=high_vif_features, errors='ignore')
X_test_final = X_test_selected.drop(columns=high_vif_features, errors='ignore')

X_vif_final = X_train_final.select_dtypes(include=[np.number]).copy()
vif_data_final = pd.DataFrame()
vif_data_final["Feature"] = X_vif_final.columns
vif_data_final["VIF"] = [variance_inflation_factor(X_vif_final.values, i)
                         for i in range(X_vif_final.shape[1])]

print("\nRecalculated VIF after pruning:")
print(vif_data_final.sort_values(by="VIF", ascending=False))

final_features = list(X_train_final.columns)
joblib.dump(final_features, 'final_features.pkl')

print("Checking final feature consistency between TRAIN and TEST sets...")
print(f"X_train_final shape: {X_train_final.shape}")
print(f"X_test_final shape: {X_test_final.shape}")
print("Final list of features to be used for model training:")
print(X_train_final.columns.tolist())


# =====================================================================
# 11. Modelling — base learners with/without ADASYN resampling
# =====================================================================
os.makedirs("saved_models", exist_ok=True)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

weight_ratio = Counter(y_train)[0] / Counter(y_train)[1]  # for XGBoost

params = {
    "lr": {'clf__C': [0.25, 0.5, 1.0], 'clf__penalty': ['l2']},
    "rf": {'clf__n_estimators': [150, 300], 'clf__max_depth': [None, 4], 'clf__min_samples_split': [5],
           'clf__min_samples_leaf': [10, 15], 'clf__max_features': [0.5], 'clf__bootstrap': [True]},
    "xgb": {'clf__learning_rate': [0.01, 0.05], 'clf__n_estimators': [150, 300], 'clf__max_depth': [3],
            'clf__subsample': [0.8, 1.0], 'clf__colsample_bytree': [0.7],
            'clf__min_child_weight': [1, 5], 'clf__reg_lambda': [1, 10]},
    "svm": [{'clf__kernel': ['linear'], 'clf__C': [0.1, 1]},
            {'clf__kernel': ['rbf'], 'clf__C': [1], 'clf__gamma': ['scale']}],
    "nb": {'clf__var_smoothing': [1e-9, 1e-10]}
}

def create_model(name, use_sample):
    if name == "Logistic Regression":
        return LogisticRegression(solver='liblinear', class_weight=None if use_sample else 'balanced')
    elif name == "Random Forest":
        return RandomForestClassifier(random_state=42, class_weight=None if use_sample else 'balanced')
    elif name == "XGBoost":
        return XGBClassifier(eval_metric='logloss', random_state=42,
                             scale_pos_weight=1 if use_sample else weight_ratio)
    elif name == "SVM":
        return SVC(probability=True, random_state=SEED, class_weight=None if use_sample else 'balanced')
    elif name == "GaussianNB":
        return GaussianNB()
    else:
        raise ValueError(f"Unknown model: {name}")

def make_pipeline_and_grid(model, param_grid, use_sample):
    steps = [('adasyn', ADASYN(random_state=42))] if use_sample else []
    steps.append(('clf', model))
    return ImbPipeline(steps), param_grid

all_model_metadata = []
best_models_per_setting = {'No Resampling': {}, 'ADASYN': {}}
best_scores_per_setting = {'No Resampling': {}, 'ADASYN': {}}
best_params_per_setting = {'No Resampling': {}, 'ADASYN': {}}
chosen_setting = {}
best_params_per_model = {}
final_base_models = {}

for tag, use_sample in [('No Resampling', False), ('ADASYN', True)]:
    print(f"\nTraining models with setting: {tag}")
    for name, grid_key in [
        ("Logistic Regression", "lr"),
        ("Random Forest", "rf"),
        ("XGBoost", "xgb"),
        ("SVM", "svm"),
        ("GaussianNB", "nb")
    ]:
        print(f" - Training {name}...")
        model = create_model(name, use_sample)
        pipe, grid = make_pipeline_and_grid(model, params[grid_key], use_sample)
        search = GridSearchCV(pipe, grid, scoring='average_precision', cv=cv, n_jobs=-1, verbose=0)
        search.fit(X_train_final, y_train)

        best_models_per_setting[tag][name] = search.best_estimator_
        best_scores_per_setting[tag][name] = search.best_score_
        best_params_per_setting[tag][name] = search.best_params_

        model_filename_all = f"saved_models/{name.replace(' ', '_')}_{tag.replace(' ', '_')}_FULL.pkl"
        joblib.dump(search.best_estimator_, model_filename_all)
        print(f"   Full model saved: {model_filename_all}")
        print(f"   Best {name} Score: {search.best_score_:.4f}")

        all_model_metadata.append({
            "model_name": name,
            "resampling": tag,
            "score": search.best_score_,
            "best_params": search.best_params_
        })

with open("saved_models/all_model_metadata.json", "w") as f:
    json.dump(all_model_metadata, f, indent=4)
print("\nAll model training metadata saved.")

print("\nBest base models and their parameters")
for name in ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "GaussianNB"]:
    setting = 'No Resampling' if best_scores_per_setting['No Resampling'][name] >= best_scores_per_setting['ADASYN'][name] else 'ADASYN'
    chosen_setting[name] = setting
    best_model = best_models_per_setting[setting][name]
    final_base_models[name] = best_model
    best_params = best_params_per_setting[setting][name]
    best_params_per_model[name] = best_params

    model_filename = f"saved_models/{name.replace(' ', '_')}_{setting.replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_filename)
    print(f" - Saved model: {name} ({setting})")
    print(f"   Best {name} params: {best_params}")

with open("saved_models/best_params_per_model.json", "w") as f:
    json.dump(best_params_per_model, f, indent=4)
print("\nBest base models and their parameters saved.")

print("\nGenerating and saving OOF predictions of the best base models...")
for name, model in final_base_models.items():
    print(f" - {name} ({chosen_setting[name]})")
    y_prob_oof = cross_val_predict(model, X_train_final, y_train, cv=cv_split, method='predict_proba', n_jobs=-1)[:, 1]
    joblib.dump(y_prob_oof, f"saved_models/y_prob_oof_{name.replace(' ', '_')}.pkl")

joblib.dump(y_train, "saved_models/y_true_train.pkl")
joblib.dump((X_train_final, y_train), "saved_models/train_data_full.pkl")
print("\nBest base models OOF predictions saved.")


# =====================================================================
# 12. Stacking Ensemble
# =====================================================================
base_model_names = ["Logistic Regression", "SVM", "Random Forest", "XGBoost", "GaussianNB"]
base_models = []

for name in base_model_names:
    best_params = best_params_per_model[name]
    setting = chosen_setting[name]
    use_sample = (setting == 'ADASYN')

    model = create_model(name, use_sample)

    if use_sample:
        steps = [('adasyn', ADASYN(random_state=42)), ('clf', model)]
        base_model = Pipeline(steps)
        base_model.set_params(**best_params)
    else:
        base_model = model
        base_model.set_params(**{key.replace('clf__', ''): val for key, val in best_params.items()})

    base_models.append((name.lower().replace(' ', '_'), base_model))

print("\nBuilding stacking model...")
meta_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=weight_ratio
)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1,
    cv=cv_split
)

meta_param_grid = {
    "final_estimator__learning_rate": [0.03],
    "final_estimator__n_estimators": [200],
    "final_estimator__max_depth": [2],
    "final_estimator__subsample": [0.8],
    "final_estimator__colsample_bytree": [0.6, 0.8],
    'final_estimator__reg_lambda': [1, 10]
}

print("Tuning meta-model with RandomizedSearchCV...")
search = RandomizedSearchCV(
    estimator=stacking_clf,
    param_distributions=meta_param_grid,
    n_iter=100,
    scoring='average_precision',
    cv=cv_split,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train_final, y_train)

best_stacking_clf = search.best_estimator_
print(f"Best meta-model params: {search.best_params_}")
print(f"Best CV PR AUC: {search.best_score_:.4f}")

pr_auc = average_precision_score(y_train, best_stacking_clf.predict_proba(X_train_final)[:, 1])
print(f"Stacking Model PR AUC on Training Data: {pr_auc:.4f}")

joblib.dump(best_stacking_clf, "saved_models/final_stacking_model.pkl")
print("Final stacking model saved successfully.")


# =====================================================================
# 13. Learning Curves
# =====================================================================
print("Generating learning curves for the best models...")

def plot_learning_curve_subplot(estimator, X, y, ax, title='', cv=None, scoring='average_precision',
                                n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    try:
        estimator = clone(estimator)
        ax.set_title(title)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("PR AUC")
        ax.grid(True)

        train_sizes, train_scores, valid_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring,
            n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True, random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                        valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax.legend(loc="best")
    except Exception as e:
        ax.set_title(f"{title} (Failed)")
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=9)
        ax.axis('off')

all_models = []
for name in final_base_models:
    setting = chosen_setting[name]
    model = final_base_models[name]
    display_name = f"{name} ({setting})"
    all_models.append((display_name, model))

all_models.append(("Stacking Ensemble", best_stacking_clf))

num_models = len(all_models)
cols = 3
rows = (num_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
axes = axes.flatten()

for i, (display_name, model) in enumerate(all_models):
    plot_learning_curve_subplot(
        model, X_train_final, y_train,
        ax=axes[i], title=display_name, cv=cv, scoring='average_precision'
    )

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("Learning_curve.png")
plt.show()
print("Learning curve generation completed.")


# =====================================================================
# 14. Test Set Evaluation
# =====================================================================
print("\nEvaluating all models on the test set...")

evaluation_results = {}
model_predictions = {}
all_models_to_evaluate = {}
resampling_settings = ['No_Resampling', 'ADASYN']
model_names = ["Logistic_Regression", "Random_Forest", "XGBoost", "SVM", "GaussianNB"]

for setting in resampling_settings:
    for name in model_names:
        filename = f"saved_models/{name}_{setting}_FULL.pkl"
        if os.path.exists(filename):
            model = joblib.load(filename)
            display_name = f"{name.replace('_', ' ')} ({setting.replace('_', ' ')})"
            all_models_to_evaluate[display_name] = model

stacking_model_path = "saved_models/final_stacking_model.pkl"
if os.path.exists(stacking_model_path):
    stacking_final = joblib.load(stacking_model_path)
    all_models_to_evaluate["Stacking Ensemble (Stacked)"] = stacking_final

for display_name, model in all_models_to_evaluate.items():
    print(f"Evaluating: {display_name}")
    y_pred = model.predict(X_test_final)

    try:
        y_proba = model.predict_proba(X_test_final)
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
    except Exception:
        y_proba = None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    evaluation_results[display_name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'PR AUC': average_precision_score(y_test, y_proba) if y_proba is not None else None,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Type II Error (FN Rate)': fn_rate
    }

    model_predictions[display_name] = {"y_pred": y_pred, "y_proba": y_proba}

# ---------------------------------------------------------------------
# Simple baseline: logistic regression on RAW features.
# No KNN imputation, outlier capping, feature engineering, feature
# selection or ADASYN. Only median imputation of the invalid zeros and
# standard scaling, both fit on the training set. Same split, same test set.
# ---------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline

baseline_name = "Baseline: Logistic Regression (raw features)"
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()
X_train_raw[zero_as_nan] = X_train_raw[zero_as_nan].replace(0, np.nan)
X_test_raw[zero_as_nan] = X_test_raw[zero_as_nan].replace(0, np.nan)

baseline_model = SkPipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced')),
])
baseline_model.fit(X_train_raw, y_train)
b_pred = baseline_model.predict(X_test_raw)
b_proba = baseline_model.predict_proba(X_test_raw)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, b_pred).ravel()
evaluation_results[baseline_name] = {
    'Precision': precision_score(y_test, b_pred),
    'Recall': recall_score(y_test, b_pred),
    'F1 Score': f1_score(y_test, b_pred),
    'ROC AUC': roc_auc_score(y_test, b_proba),
    'PR AUC': average_precision_score(y_test, b_proba),
    'Accuracy': accuracy_score(y_test, b_pred),
    'Balanced Accuracy': balanced_accuracy_score(y_test, b_pred),
    'MCC': matthews_corrcoef(y_test, b_pred),
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    'Type II Error (FN Rate)': fn / (fn + tp) if (fn + tp) > 0 else 0,
}
model_predictions[baseline_name] = {"y_pred": b_pred, "y_proba": b_proba}

results_df = pd.DataFrame(evaluation_results).T.reset_index().rename(columns={'index': 'Model'})

summary_cols = ['Model', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC', 'Accuracy', 'Balanced Accuracy', 'MCC']
summary_df = results_df[summary_cols].sort_values(by='Recall', ascending=False)
print("\nTest Set Results Summary:")
print(summary_df.to_string(index=False))

confusion_cols = ['Model', 'TN', 'FP', 'FN', 'TP', 'Type II Error (FN Rate)']
confusion_df = results_df[confusion_cols].set_index('Model').sort_values(by='Type II Error (FN Rate)', ascending=True)
print("\nTest Set Confusion Matrices:")
print(confusion_df.to_string())

os.makedirs("saved_evaluations", exist_ok=True)
joblib.dump(results_df, "saved_evaluations/evaluation_results_df.pkl")
joblib.dump(summary_df, "saved_evaluations/summary_df.pkl")
joblib.dump(confusion_df, "saved_evaluations/confusion_df.pkl")
joblib.dump(model_predictions, "saved_evaluations/model_predictions.pkl")
joblib.dump({"X_test_final": X_test_final, "y_test": y_test}, "saved_evaluations/test_data.pkl")
print("\nAll evaluation results and predictions have been saved as pickle files in 'saved_evaluations/' directory.")

# Small human-readable results for the repo (test set, held-out)
os.makedirs("results", exist_ok=True)
summary_df.round(4).to_csv("results/test_set_metrics.csv", index=False)
confusion_df.round(4).to_csv("results/test_set_confusion_matrices.csv")
import sklearn, xgboost, imblearn
with open("results/run_info.json", "w") as f:
    json.dump({
        "random_seed": SEED,
        "n_train": int(len(X_train_final)),
        "n_test": int(len(X_test_final)),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "final_features": list(X_train_final.columns),
        "python_packages": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                            "imbalanced-learn": imblearn.__version__, "shap": shap.__version__,
                            "pandas": pd.__version__, "numpy": np.__version__},
    }, f, indent=2)
print("Test-set metrics exported to results/*.csv")


# =====================================================================
# 15. Training / Inference Timing
# =====================================================================
print("\nEmpirical Training and Inference Times for All Models:\n")
timing_results = []

for setting in ['No Resampling', 'ADASYN']:
    for name, model in best_models_per_setting[setting].items():
        model_name = f"{name} ({setting})"
        model_copy = clone(model)
        try:
            start_train = time.perf_counter()
            model_copy.fit(X_train_final, y_train)
            end_train = time.perf_counter()
            training_duration = end_train - start_train

            start_infer = time.perf_counter()
            _ = model_copy.predict(X_test_final)
            end_infer = time.perf_counter()
            inference_duration = end_infer - start_infer

            timing_results.append({
                'Model': model_name,
                'Training Time (s)': round(training_duration, 4),
                'Inference Time (s)': round(inference_duration, 4)
            })
        except Exception as e:
            print(f" Failed timing for {model_name}: {e}")
            timing_results.append({'Model': model_name, 'Training Time (s)': None, 'Inference Time (s)': None})

try:
    stacking_clone = clone(stacking_final)
    start_train = time.perf_counter()
    stacking_clone.fit(X_train_final, y_train)
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    _ = stacking_clone.predict(X_test_final)
    end_infer = time.perf_counter()

    timing_results.append({
        'Model': 'Stacking Ensemble (Stacked)',
        'Training Time (s)': round(end_train - start_train, 4),
        'Inference Time (s)': round(end_infer - start_infer, 4)
    })
except Exception as e:
    print(f" Failed timing for Stacking Ensemble: {e}")
    timing_results.append({'Model': 'Stacking Ensemble (Stacked)', 'Training Time (s)': None, 'Inference Time (s)': None})

timing_df = pd.DataFrame(timing_results).sort_values(by='Training Time (s)', na_position='last')
print("Training and Inference Time Summary:\n")
print(timing_df.to_string(index=False))


# =====================================================================
# 16. Overfitting / Underfitting Diagnostics
# =====================================================================
print("\nCross-validated Train vs Test PR AUC with Overfitting Status:")

OVERFITTING_THRESHOLD = 0.1
UNDERFITTING_THRESHOLD = 0.6

def get_fit_status(train_auc, test_auc):
    gap = train_auc - test_auc
    if gap > OVERFITTING_THRESHOLD:
        return "Overfitting"
    elif test_auc < UNDERFITTING_THRESHOLD:
        return "Underfitting"
    else:
        return "Good Fit"

cv_auc_results = {}

for name, model in all_models_to_evaluate.items():
    print(f"Evaluating: {name}")
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        print(f" Skipping {name} (no prob/decision output)")
        continue

    try:
        train_auc_scores = cross_val_score(model, X_train_final, y_train,
                                           scoring='average_precision', cv=cv_split, n_jobs=-1)
        train_mean_auc = np.mean(train_auc_scores)

        model.fit(X_train_final, y_train)

        if hasattr(model, "predict_proba"):
            y_test_scores = model.predict_proba(X_test_final)[:, 1]
        else:
            y_test_scores = model.decision_function(X_test_final)

        test_auc = average_precision_score(y_test, y_test_scores)  # PR AUC, to match the CV PR AUC column

        cv_auc_results[name] = {
            'CV PR AUC (Train)': round(train_mean_auc, 3),
            'Test PR AUC': round(test_auc, 3),
            'Overfitting Gap': round(train_mean_auc - test_auc, 3),
            'Fit Status': get_fit_status(train_mean_auc, test_auc)
        }
    except Exception as e:
        print(f" Failed on {name}: {e}")
        cv_auc_results[name] = {
            'CV PR AUC (Train)': None, 'Test PR AUC': None,
            'Overfitting Gap': None, 'Fit Status': "Error"
        }

cv_auc_df = pd.DataFrame(cv_auc_results).T.sort_values(by='Overfitting Gap', ascending=True)
print("\nCross-validated Train vs Test PR AUC Summary:")
print(cv_auc_df.to_string())

cv_auc_df_sorted = cv_auc_df.sort_values(by="Test PR AUC", ascending=False)

def get_test_bar_color(status):
    if status == "Overfitting":
        return "red"
    elif status == "Underfitting":
        return "gold"
    else:
        return "green"

plt.figure(figsize=(12, 6))
bar_width = 0.4
indices = np.arange(len(cv_auc_df_sorted))
test_bar_colors = [get_test_bar_color(status) for status in cv_auc_df_sorted['Fit Status']]

plt.bar(indices - bar_width / 2, cv_auc_df_sorted['CV PR AUC (Train)'],
        width=bar_width, label='Train PR AUC', color='skyblue')
plt.bar(indices + bar_width / 2, cv_auc_df_sorted['Test PR AUC'],
        width=bar_width, label='Test PR AUC', color=test_bar_colors)

for i, (model_name, row) in enumerate(cv_auc_df_sorted.iterrows()):
    if row['Fit Status'] == "Overfitting":
        plt.text(i + bar_width / 2, row['Test PR AUC'] + 0.02, 'Overfitting',
                 ha='center', va='bottom', color='red', fontsize=8, fontweight='bold')

plt.xticks(indices, cv_auc_df_sorted.index, rotation=45, ha='right')
plt.ylabel("PR AUC Score")
plt.title("Train vs. Test PR AUC with Fit Status")
plt.ylim(0, 1.05)
plt.axhline(0.6, color='gray', linestyle='--', linewidth=0.8, label='Underfitting Threshold')
plt.legend()
plt.tight_layout()
plt.savefig("saved_models/train_vs_test_roc_auc_colored.png", dpi=300)
plt.show()


# =====================================================================
# 17. Model Evaluation Dashboard
# =====================================================================
print("Generating Model Evaluation Dashboard...")
results_df = joblib.load("saved_evaluations/evaluation_results_df.pkl")

def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

weights = {'Recall': 0.4, 'PR AUC': 0.3, 'F1 Score': 0.2, 'MCC': 0.1}

results_df['Custom_Score'] = (
    results_df['Recall'] * weights['Recall'] +
    results_df['PR AUC'] * weights['PR AUC'] +
    results_df['F1 Score'] * weights['F1 Score'] +
    results_df['MCC'] * weights['MCC']
)

results_df['Recall_norm'] = min_max_norm(results_df['Recall'])
results_df['MCC_norm'] = min_max_norm(results_df['MCC'])
results_df['F1_norm'] = min_max_norm(results_df['F1 Score'])
results_df['PR_AUC_norm'] = min_max_norm(results_df['PR AUC'])

ranked_df = results_df.sort_values(by='Custom_Score', ascending=False)

stacked_data = ranked_df[['Model', 'Recall_norm', 'MCC_norm', 'F1_norm', 'PR_AUC_norm']].copy()
stacked_data.set_index('Model', inplace=True)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(data=ranked_df, x='Model', y='Custom_Score', palette='viridis', ax=ax1)
ax1.set_title("Custom Score by Model", fontsize=14)
ax1.set_ylabel("Custom Score (weighted sum of raw metrics)")
ax1.set_xlabel("")
ax1.set_ylim(0, ranked_df['Custom_Score'].max() + 0.05)
ax1.set_xticklabels(ranked_df['Model'], rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax1.patches:
    ax1.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.005, f"{p.get_height():.3f}",
             ha='center', va='bottom', fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
stacked_data.plot(kind='bar', stacked=True, ax=ax2, colormap='Set2', legend=False)
ax2.set_title("Metric Contribution to Custom Score (Normalized)", fontsize=14)
ax2.set_ylabel("Normalized Metric Value")
ax2.set_xlabel("")
ax2.set_ylim(0, 4.1)
ax2.set_xticklabels(stacked_data.index, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')

ax3 = fig.add_subplot(gs[1, :])
sns.heatmap(
    stacked_data, annot=True, fmt=".2f", cmap="YlGnBu",
    cbar_kws={"label": "Normalized Score"}, ax=ax3
)
ax3.set_title("Normalized Metric Scores (Heatmap)", fontsize=14)
ax3.set_ylabel("Model")
ax3.set_xlabel("Metric")

plt.suptitle("Model Evaluation Dashboard", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Model Evaluation Dashboard.png")
plt.show()
print("Dashboard generation completed.")


# =====================================================================
# 18. ROC and Precision-Recall Curves for the Best Six Models
# =====================================================================
print("Generating ROC curves for the best six models...")

selected_models = [
    'Stacking Ensemble (Stacked)',
    'Random Forest (No Resampling)',
    'XGBoost (No Resampling)',
    'XGBoost (ADASYN)',
    'Random Forest (ADASYN)',
    'SVM (ADASYN)',
]

n_models = len(selected_models)
n_cols = 3
n_rows = -(-n_models // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

for i, model_name in enumerate(selected_models):
    if model_name not in all_models_to_evaluate:
        print(f"Warning: Model '{model_name}' not found. Skipping.")
        continue

    model = all_models_to_evaluate[model_name]
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_final)
            y_scores = proba[:, 1] if proba.ndim > 1 else proba
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test_final)
        else:
            print(f"Warning: Model '{model_name}' does not support scoring. Skipping.")
            continue
    except Exception as e:
        print(f"Error getting scores for '{model_name}': {e}")
        continue

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)

    ax = axes[i]
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], color='red', linestyle='dashed', label='Chance (AUC = 0.5)')
    ax.set_title(model_name, fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("ROC Curves for best six Models", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("saved_evaluations/roc_curves.png", bbox_inches='tight')
plt.show()
print("ROC curve generation completed.")

print("Generating PR curves for selected models...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()
positive_rate = y_test.mean()

for i, model_name in enumerate(selected_models):
    if model_name not in all_models_to_evaluate:
        print(f"Warning: Model '{model_name}' not found. Skipping.")
        continue

    model = all_models_to_evaluate[model_name]
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_final)
            y_scores = proba[:, 1] if proba.ndim > 1 else proba
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test_final)
        else:
            print(f"Warning: Model '{model_name}' does not support scoring. Skipping.")
            continue
    except Exception as e:
        print(f"Error getting scores for '{model_name}': {e}")
        continue

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    ax = axes[i]
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.2f}")
    ax.fill_between(recall, precision, step='post', alpha=0.2)
    ax.hlines(positive_rate, 0, 1, colors='red', linestyles='dashed', label=f'Baseline = {positive_rate:.2f}')
    ax.set_title(model_name, fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.5)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Precision-Recall Curves for best six Models", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("saved_evaluations/pr_curves.png", bbox_inches='tight')
plt.show()
print("PR curve generation completed.")


# =====================================================================
# 19. SHAP — Stacking Ensemble Meta-Model
# =====================================================================
print("Generating SHAP plot to check feature attribution to the ensemble...")

meta_X = []
for name in base_model_names:
    oof_preds = joblib.load(f"saved_models/y_prob_oof_{name.replace(' ', '_')}.pkl")
    meta_X.append(oof_preds)

meta_X = np.column_stack(meta_X)
meta_X_df = pd.DataFrame(meta_X, columns=[f"{name}_oof" for name in base_model_names])

X_train_final, y_train = joblib.load("saved_models/train_data_full.pkl")

# Column order must match what the meta-model was trained on:
# base-model predictions first, then the passthrough input features (real names kept).
meta_input = pd.concat([meta_X_df.reset_index(drop=True),
                        pd.DataFrame(X_train_final).reset_index(drop=True)], axis=1)

explainer = shap.TreeExplainer(best_stacking_clf.final_estimator_)
shap_values = explainer(meta_input)

plt.close('all')
shap.summary_plot(shap_values, meta_input, plot_type="bar", show=False)
plt.title("\nContribution to Stacking Ensemble")
plt.xlabel("Mean |SHAP value| (Feature importance)")
plt.tight_layout()
plt.savefig("figures/Contribution_to_Stacking_Ensemble.png", dpi=300, bbox_inches='tight')

joblib.dump(shap_values, "saved_models/meta_model_shap_values.pkl")


# =====================================================================
# 20. SHAP — Best Two Base Models
# =====================================================================
print("Generating SHAP explanations...")

X_sample = X_test_final.sample(n=100, random_state=42)

xgb_clf = all_models_to_evaluate['XGBoost (ADASYN)'].named_steps['clf']
explainer_xgb = shap.Explainer(xgb_clf)
shap_values_xgb = explainer_xgb(X_sample)

plt.close('all')
shap.summary_plot(shap_values_xgb, X_sample, show=False)
plt.title("XGBoost SHAP Summary")
plt.tight_layout()
plt.savefig("figures/shap_summary_xgboost.png", dpi=300)

shap.initjs()

force_plot_xgb = shap.force_plot(
    explainer_xgb.expected_value,
    shap_values_xgb.values[0, :],
    X_sample.iloc[0]
)
shap.save_html("force_plot_xgboost.html", force_plot_xgb)
display(force_plot_xgb)
print("SHAP generation completed.")

print("Generating SHAP explanations...")
rf_clf = all_models_to_evaluate['Random Forest (ADASYN)'].named_steps['clf']
X_sample_rf = X_sample.copy()
explainer_rf = shap.TreeExplainer(rf_clf)
shap_values_rf = explainer_rf.shap_values(X_sample_rf)

plt.close('all')
shap.summary_plot(shap_values_rf[:, :, 1], X_sample_rf, show=False)
plt.title("Random Forest SHAP Summary (Class 1)")
plt.tight_layout()
plt.savefig("figures/shap_summary_random_forest.png", dpi=300)

force_plot_rf = shap.force_plot(
    explainer_rf.expected_value[1],
    shap_values_rf[0, :, 1],
    X_sample_rf.iloc[0]
)
shap.save_html("force_plot_random_forest.html", force_plot_rf)
display(force_plot_rf)
print("SHAP generation completed.")


# =====================================================================
# 21. Feature Importance (Mean |SHAP|) — Best Two Base Models
# =====================================================================
xgb_mean_abs_shap = np.abs(shap_values_xgb.values).mean(axis=0)
xgb_importance_df = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean |SHAP|': xgb_mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)

rf_mean_abs_shap = np.abs(shap_values_rf[:, :, 1]).mean(axis=0)
rf_importance_df = pd.DataFrame({
    'Feature': X_sample_rf.columns,
    'Mean |SHAP|': rf_mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)

top_n = 10
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

axes[0].barh(
    xgb_importance_df['Feature'].head(top_n)[::-1],
    xgb_importance_df['Mean |SHAP|'].head(top_n)[::-1],
    color='tomato'
)
axes[0].set_title("XGBoost (ADASYN)", fontsize=14)
axes[0].set_xlabel("Mean |SHAP value|")
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].barh(
    rf_importance_df['Feature'].head(top_n)[::-1],
    rf_importance_df['Mean |SHAP|'].head(top_n)[::-1],
    color='seagreen'
)
axes[1].set_title("Random Forest (ADASYN)", fontsize=14)
axes[1].set_xlabel("Mean |SHAP value|")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle("Feature Importance (Mean Absolute SHAP) Comparison", fontsize=16)
plt.tight_layout()
plt.show()


# =====================================================================
# 22. Partial Dependence Plots — XGBoost (ADASYN)
# =====================================================================
print("\nGenerating Partial Dependence Plots for best model (XGBoost - ADASYN)...")

xgb_clf = all_models_to_evaluate['XGBoost (ADASYN)'].named_steps['clf']
features_to_plot = ['Glucose', 'Insulin', 'BMI', 'AgeGroup', 'Pregnancies', 'BloodPressure']

pdp_disp = PartialDependenceDisplay.from_estimator(
    xgb_clf, X_test_final, features_to_plot,
    kind='average', grid_resolution=50, n_jobs=-1
)

fig = pdp_disp.figure_
fig.set_size_inches(15, 10)
fig.suptitle("Partial Dependence Plots - XGBoost (ADASYN)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

for ax in pdp_disp.axes_.ravel():
    if ax is not None:
        ax.grid(True)

plt.savefig("figures/Partial_Dependence_Plots.png", dpi=300, bbox_inches='tight')
plt.show()

print("PDPs generation for top features completed.")


# =====================================================================
# 23. Final comparison figure (test set) and curated figures/ folder
# =====================================================================
os.makedirs("figures", exist_ok=True)
compare = [baseline_name, "Stacking Ensemble (Stacked)", "XGBoost (ADASYN)", "Random Forest (ADASYN)"]

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
for m in compare:
    if m not in model_predictions or model_predictions[m]["y_proba"] is None:
        continue
    p = model_predictions[m]["y_proba"]
    fpr, tpr, _ = roc_curve(y_test, p)
    prec, rec, _ = precision_recall_curve(y_test, p)
    short = m.replace("Baseline: Logistic Regression (raw features)", "Baseline (LR, raw features)")
    ax_roc.plot(fpr, tpr, lw=2, label=f"{short} (AUC {roc_auc_score(y_test, p):.2f})")
    ax_pr.plot(rec, prec, lw=2, label=f"{short} (AP {average_precision_score(y_test, p):.2f})")
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax_pr.hlines(y_test.mean(), 0, 1, colors="k", linestyles="--", lw=1, label=f"Prevalence ({y_test.mean():.2f})")
ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (held-out test set)")
ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (held-out test set)")
for a in (ax_roc, ax_pr):
    a.grid(alpha=0.3)
    a.legend(loc="lower right" if a is ax_roc else "upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/roc_pr_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Key figures saved to figures/")
