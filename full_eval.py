import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from health_model import compute_health

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "wear",
    "Machine failure": "failure",
})

results = df.apply(lambda r: compute_health({
    "air_temp": r["air_temp"], "process_temp": r["process_temp"],
    "rpm": r["rpm"], "torque": r["torque"], "wear": r["wear"], "Type": r["Type"],
}), axis=1, result_type="expand")

df = pd.concat([df, results], axis=1)
df.to_csv("data/scored_dataset.csv", index=False)

print("=" * 70)
print("HEALTH SCORE DISTRIBUTION -- full 10,000 rows")
print("=" * 70)
print(df["health_score"].describe())

print("\n--- histogram (10-point buckets) ---")
bins = list(range(0, 101, 10))
df["health_bucket"] = pd.cut(df["health_score"], bins=bins, include_lowest=True)
bucket_table = df.groupby("health_bucket", observed=True).agg(
    count=("health_score", "size"),
    actual_failures=("failure", "sum"),
)
bucket_table["failure_rate_in_bucket"] = (bucket_table["actual_failures"] / bucket_table["count"]).round(4)
print(bucket_table)

print("\n--- health score distribution split by actual outcome ---")
print(df.groupby("failure")["health_score"].describe()[["count", "mean", "50%", "min", "max"]])

print("\n" + "=" * 70)
print("THRESHOLD CURVE -- 'predict failure if health_score < threshold'")
print("=" * 70)
risk_score = 100 - df["health_score"]  # higher = more at-risk
y_true = df["failure"].values

auc = roc_auc_score(y_true, risk_score)
ap = average_precision_score(y_true, risk_score)
print(f"\nROC-AUC: {auc:.4f}")
print(f"Average Precision (area under PR curve): {ap:.4f}")

rows = []
for threshold in [95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10, 5, 1]:
    pred_positive = df["health_score"] < threshold
    tp = (pred_positive & (df["failure"] == 1)).sum()
    fp = (pred_positive & (df["failure"] == 0)).sum()
    fn = (~pred_positive & (df["failure"] == 1)).sum()
    tn = (~pred_positive & (df["failure"] == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    rows.append({
        "health_threshold": threshold,
        "flagged_count": int(pred_positive.sum()),
        "flagged_pct_of_fleet": round(pred_positive.mean() * 100, 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
    })

curve_df = pd.DataFrame(rows)
print(curve_df.to_string(index=False))

print("\n--- all 339 actual failures: what health score did the model give them? ---")
fail_rows = df[df["failure"] == 1]
print(fail_rows["health_score"].describe())
print("\ncount of actual failures the model scored as 'healthy' (health_score >= 70):")
missed_high = fail_rows[fail_rows["health_score"] >= 70]
print(f"{len(missed_high)} / {len(fail_rows)}")
if len(missed_high) > 0:
    print(missed_high[["UDI", "Type", "torque", "wear", "health_score", "dominant_mode"]].to_string(index=False) if len(missed_high) <= 30 else missed_high[["UDI","Type","torque","wear","health_score","dominant_mode"]].head(30).to_string(index=False))
    print("\ndominant_mode breakdown of these misses:")
    print(missed_high["dominant_mode"].value_counts())
