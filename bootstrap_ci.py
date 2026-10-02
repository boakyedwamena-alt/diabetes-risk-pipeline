"""95% bootstrap confidence intervals on the held-out test set (baseline vs stacking ensemble).

Run AFTER diabetes_pipeline.py (needs saved_evaluations/*.pkl). Writes results/bootstrap_ci.csv.
The test set is small (154 patients, 54 positive), so single-number comparisons are noisy.
"""
import joblib, numpy as np, pandas as pd, warnings
from sklearn.metrics import recall_score, average_precision_score, roc_auc_score, accuracy_score
warnings.filterwarnings("ignore")

preds = joblib.load("saved_evaluations/model_predictions.pkl")
y = joblib.load("saved_evaluations/test_data.pkl")["y_test"].values
BASE = "Baseline: Logistic Regression (raw features)"
STACK = "Stacking Ensemble (Stacked)"
metrics = {
    "Recall": lambda yt, p, s: recall_score(yt, p),
    "PR AUC": lambda yt, p, s: average_precision_score(yt, s),
    "ROC AUC": lambda yt, p, s: roc_auc_score(yt, s),
    "Accuracy": lambda yt, p, s: accuracy_score(yt, p),
}
rng = np.random.default_rng(42)
draws = {m: {k: [] for k in metrics} for m in (BASE, STACK)}
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    if y[idx].sum() == 0:
        continue
    for m in (BASE, STACK):
        for k, f in metrics.items():
            draws[m][k].append(f(y[idx], preds[m]["y_pred"][idx], preds[m]["y_proba"][idx]))

rows = []
for k in metrics:
    b, s = np.array(draws[BASE][k]), np.array(draws[STACK][k])
    d = s - b
    rows.append({"Metric": k,
                 "Baseline 95% CI": f"[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]",
                 "Stacking 95% CI": f"[{np.percentile(s,2.5):.3f}, {np.percentile(s,97.5):.3f}]",
                 "Stacking - Baseline (mean)": round(d.mean(), 3),
                 "Difference 95% CI": f"[{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]"})
out = pd.DataFrame(rows)
out.to_csv("results/bootstrap_ci.csv", index=False)
print(out.to_string(index=False))
