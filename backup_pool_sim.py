"""
Backup-pool sizing + cost-tradeoff simulation.

Carries the FULL health-model precision/recall curve (data/scored_dataset.csv,
thresholds swept 1-99) into an economic model, rather than locking a single
"best" threshold in advance. For each threshold:

    total_expected_cost = FP * inspection_pull_cost
                         + FN * (repair_time_hours * downtime_cost_per_hour)
                         + pool_size * holding_cost_per_spare

Design notes / assumptions (see PARAMS docstring below for what's cited vs.
what's a flagged assumption):

  - FP cost is a flat "unnecessary pull" cost, deliberately NOT scaled by the
    downtime-cost benchmark. Pulling a robot for a quick check that turns out
    fine is a routine planned action (labor + brief inspection), not a
    production-outage event -- a fundamentally different cost driver than a
    missed failure. This also lets peak-season scaling show up honestly: FN
    cost rises with the seasonal downtime benchmark, FP cost does not, so the
    asymmetry that should push thresholds tighter in peak season is actually
    present in the model (if both scaled together, the ratio between them
    would stay ~constant and the "seasonal flex" would be a non-effect).

  - Backup-pool SIZE (for capacity/holding-cost purposes) is driven by
    Little's Law: avg_robots_down = ((TP+FN)*repair_hours + FP*inspection_hours)
    / period_hours. TP is included here (a true positive still consumes a
    real repair slot) even though it isn't a separately *monetized* error
    cost in the total_cost formula above -- a correct catch is the system
    working as intended, not an error, but it still occupies pool capacity.
    Pool size is then the smallest capacity covering a target service level
    against Poisson-distributed concurrent-down demand (same logic as a
    safety-stock reorder point under Poisson demand).

  - period_hours=24 and FLEET_SIZE=10,000 (=# of dataset rows) are a
    STRUCTURAL assumption bridging the AI4I dataset's cross-sectional
    snapshot into a flow-rate model: we treat the 10,000-row sample as one
    day's full-fleet monitoring pass. The AI4I failure incidence (3.39%) is
    a synthetic industrial benchmark, not a real robot fleet's failure rate
    -- this whole simulation is a mechanics demonstration (Little's Law +
    Poisson service level + threshold cost sweep), not a literal prediction
    about Amazon's fleet. Swap in real cadence/incidence data to make it one.

IMPORTANT: threshold=70, used earlier purely to enumerate a "how bad can the
false-negative set possibly be" diagnostic ceiling, is NOT treated here as a
candidate operating point. It has no special status in this sweep.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

df = pd.read_csv("data/scored_dataset.csv")
FLEET_SIZE = len(df)

PARAMS = {
    # --- cited benchmarks (adjustable) ---
    "repair_time_hours": 2.5,                      # industry benchmark ~1-4 hrs, midpoint used as default
    "downtime_cost_per_hour_baseline": 100_000,     # low end of cited $100K-$260K/hr range
    "downtime_cost_per_hour_peak": 260_000,         # high end of the SAME cited range, used directly as the peak-season scenario

    # --- structural assumption bridging cross-sectional data into a flow model ---
    "period_hours": 24,                             # one full-fleet monitoring pass per day

    # --- flagged assumptions, not cited, fully adjustable ---
    "inspection_pull_cost": 500,                    # flat cost of an unnecessary pull/check (labor + logistics), independent of downtime-cost benchmark
    "inspection_downtime_hours": 0.5,                # time a pulled robot is offline for a check -- used ONLY for pool-capacity sizing, not directly monetized
    "holding_cost_per_spare_per_period": 100,        # capital/depreciation + storage cost per spare robot per day
    "target_service_level": 0.95,                    # pool sized to cover this fraction of days without a shortfall
}


def pool_size_for_service_level(avg_down, service_level):
    if avg_down <= 0:
        return 0
    k = 0
    while poisson.cdf(k, avg_down) < service_level:
        k += 1
    return k


def evaluate_threshold(threshold, downtime_cost_per_hour, p=PARAMS):
    pred_positive = df["health_score"] < threshold
    tp = int((pred_positive & (df["failure"] == 1)).sum())
    fp = int((pred_positive & (df["failure"] == 0)).sum())
    fn = int((~pred_positive & (df["failure"] == 1)).sum())
    tn = int((~pred_positive & (df["failure"] == 0)).sum())

    fp_cost_total = fp * p["inspection_pull_cost"]
    fn_cost_total = fn * p["repair_time_hours"] * downtime_cost_per_hour

    avg_down = ((tp + fn) * p["repair_time_hours"] + fp * p["inspection_downtime_hours"]) / p["period_hours"]
    pool_size = pool_size_for_service_level(avg_down, p["target_service_level"])
    pool_cost = pool_size * p["holding_cost_per_spare_per_period"]

    total_cost = fp_cost_total + fn_cost_total + pool_cost

    return {
        "threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else np.nan,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else np.nan,
        "avg_robots_down": round(avg_down, 1),
        "pool_size": pool_size,
        "fp_cost": round(fp_cost_total),
        "fn_cost": round(fn_cost_total),
        "pool_cost": round(pool_cost),
        "total_cost": round(total_cost),
    }


thresholds = list(range(1, 100, 1))

scenarios = {
    "BASELINE": PARAMS["downtime_cost_per_hour_baseline"],
    "PEAK_SEASON": PARAMS["downtime_cost_per_hour_peak"],
}

curves = {}
for label, dtc in scenarios.items():
    rows = [evaluate_threshold(t, dtc) for t in thresholds]
    curves[label] = pd.DataFrame(rows)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

for label, curve in curves.items():
    dtc = scenarios[label]
    print("=" * 70)
    print(f"{label}  (downtime_cost_per_hour = ${dtc:,})")
    print("=" * 70)
    # print a readable subset -- every 5th threshold, plus the minimum
    display_rows = curve[curve["threshold"] % 5 == 0].copy()
    best_idx = curve["total_cost"].idxmin()
    best_row = curve.loc[[best_idx]]
    display_rows = pd.concat([display_rows, best_row]).drop_duplicates(subset="threshold").sort_values("threshold")
    print(display_rows.to_string(index=False))
    print(f"\n>>> COST-MINIMIZING THRESHOLD: {curve.loc[best_idx,'threshold']}  "
          f"(total_cost=${curve.loc[best_idx,'total_cost']:,}, "
          f"recall={curve.loc[best_idx,'recall']:.1%}, precision={curve.loc[best_idx,'precision']:.1%}, "
          f"pool_size={curve.loc[best_idx,'pool_size']})")
    print()

print("=" * 70)
print("SEASONAL SHIFT SUMMARY")
print("=" * 70)
base_best = curves["BASELINE"].loc[curves["BASELINE"]["total_cost"].idxmin()]
peak_best = curves["PEAK_SEASON"].loc[curves["PEAK_SEASON"]["total_cost"].idxmin()]
print(f"Baseline optimal threshold:    {base_best['threshold']:.0f}  (recall={base_best['recall']:.1%}, pool_size={base_best['pool_size']:.0f})")
print(f"Peak-season optimal threshold: {peak_best['threshold']:.0f}  (recall={peak_best['recall']:.1%}, pool_size={peak_best['pool_size']:.0f})")
print(f"Shift: {peak_best['threshold'] - base_best['threshold']:+.0f} threshold points, "
      f"{peak_best['recall'] - base_best['recall']:+.2%} recall, "
      f"{peak_best['pool_size'] - base_best['pool_size']:+.0f} pool slots")

curves["BASELINE"].to_csv("data/cost_curve_baseline.csv", index=False)
curves["PEAK_SEASON"].to_csv("data/cost_curve_peak.csv", index=False)
