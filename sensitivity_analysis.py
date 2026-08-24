"""
Two sensitivity checks on the shortfall-gated joint (threshold, pool_size)
optimum, per grounding feedback:

  1. Holding cost was likely overestimated at the $100/day placeholder.
     Standard equipment holding-cost practice: ~20-30%/year of asset value.
     For a $25K-$80K robot: 0.20*25000/365 = $13.7/day  ...  0.30*80000/365 = $65.8/day
     -> swept range $15-65/day, replacing the flat $100 assumption.

  2. Swap cost ($1,500) is grounded (not sourced) via: warehouse AMR annual
     maintenance $2,000-$8,000/robot, ~60% preventive / 40% unplanned ->
     $800-$3,200 per unplanned repair incident (assuming ~1 incident/robot/yr).
     $1,500 sits inside that range. Light sensitivity sweep $500-$5,000 to
     check whether the "favor pool coverage over blanket inspection" policy
     conclusion is robust across the whole plausible range, or fragile.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

df = pd.read_csv("data/scored_dataset.csv")

BASE_PARAMS = {
    "repair_time_hours": 2.5,
    "period_hours": 24,
    "inspection_pull_cost": 500,
    "inspection_downtime_hours": 0.5,
}

thresholds = list(range(1, 100, 1))
pool_sizes = list(range(0, 301, 1))


def threshold_flow(threshold):
    pred_positive = df["health_score"] < threshold
    tp = int((pred_positive & (df["failure"] == 1)).sum())
    fp = int((pred_positive & (df["failure"] == 0)).sum())
    fn = int((~pred_positive & (df["failure"] == 1)).sum())
    return tp, fp, fn


def joint_optimum(downtime_cost_per_hour, holding_cost_per_spare, swap_cost):
    full_fn_cost = BASE_PARAMS["repair_time_hours"] * downtime_cost_per_hour
    pool_arr = np.array(pool_sizes)
    best = None
    for t in thresholds:
        tp, fp, fn = threshold_flow(t)
        avg_down = ((tp + fn) * BASE_PARAMS["repair_time_hours"] + fp * BASE_PARAMS["inspection_downtime_hours"]) / BASE_PARAMS["period_hours"]
        shortfall_prob_arr = 1 - poisson.cdf(pool_arr, avg_down) if avg_down > 0 else np.zeros_like(pool_arr, dtype=float)
        expected_cost_per_fn_arr = shortfall_prob_arr * full_fn_cost + (1 - shortfall_prob_arr) * swap_cost
        fn_cost_arr = fn * expected_cost_per_fn_arr
        fp_cost_total = fp * BASE_PARAMS["inspection_pull_cost"]
        pool_cost_arr = pool_arr * holding_cost_per_spare
        total_cost_arr = fp_cost_total + fn_cost_arr + pool_cost_arr
        idx = int(np.argmin(total_cost_arr))
        row = {
            "threshold": t, "pool_size": pool_sizes[idx], "total_cost": total_cost_arr[idx],
            "shortfall_prob": shortfall_prob_arr[idx], "FN": fn, "FP": fp, "avg_down": avg_down,
        }
        if best is None or row["total_cost"] < best["total_cost"]:
            best = row
    return best


pd.set_option("display.width", 160)

print("=" * 70)
print("SENSITIVITY 1 -- holding cost swept $15-65/day (baseline downtime cost, swap=$1500)")
print("=" * 70)
print("Grounding: 20-30%/yr of a $25K-$80K robot's asset value / 365 days")
print(f"  low:  0.20 x $25,000 / 365 = ${0.20*25000/365:.1f}/day")
print(f"  high: 0.30 x $80,000 / 365 = ${0.30*80000/365:.1f}/day")
print()

rows1 = []
for hc in range(15, 66, 5):
    best = joint_optimum(downtime_cost_per_hour=100_000, holding_cost_per_spare=hc, swap_cost=1500)
    rows1.append({
        "holding_cost_per_spare_per_day": hc,
        "optimal_threshold": best["threshold"],
        "optimal_pool_size": best["pool_size"],
        "shortfall_prob": round(best["shortfall_prob"], 5),
        "total_cost": round(best["total_cost"]),
    })
curve1 = pd.DataFrame(rows1)
print(curve1.to_string(index=False))

# also show the old $100/day placeholder for direct before/after comparison
old_best = joint_optimum(downtime_cost_per_hour=100_000, holding_cost_per_spare=100, swap_cost=1500)
print(f"\n(for comparison, old $100/day placeholder: threshold={old_best['threshold']}, "
      f"pool_size={old_best['pool_size']}, total_cost=${old_best['total_cost']:,.0f})")

print("\n" + "=" * 70)
print("SENSITIVITY 2 -- swap cost swept $500-$5,000 (baseline downtime cost, holding=$40/day midpoint)")
print("=" * 70)

rows2 = []
for sc in range(500, 5001, 500):
    best = joint_optimum(downtime_cost_per_hour=100_000, holding_cost_per_spare=40, swap_cost=sc)
    rows2.append({
        "swap_cost": sc,
        "optimal_threshold": best["threshold"],
        "optimal_pool_size": best["pool_size"],
        "shortfall_prob": round(best["shortfall_prob"], 5),
        "total_cost": round(best["total_cost"]),
    })
curve2 = pd.DataFrame(rows2)
print(curve2.to_string(index=False))

thresholds_seen = sorted(curve2["optimal_threshold"].unique())
print(f"\ndistinct optimal thresholds across the full $500-$5,000 swap-cost range: {thresholds_seen}")
if len(thresholds_seen) == 1 and thresholds_seen[0] == 1:
    print("-> threshold never leaves the low-flagging regime (stays pinned at 1); only pool_size responds.")
else:
    print("-> threshold DOES move within this range -- policy conclusion is sensitive to swap cost.")
