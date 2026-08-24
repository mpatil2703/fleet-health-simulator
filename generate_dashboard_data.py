"""
Generates the saved output files the Streamlit dashboard reads from.
Reuses the SAME validated functions built earlier in this project
(reassignment_sim_v2.run_simulation, the corrected shortfall-gated joint
threshold/pool-size cost logic from cost_model_v2.py) -- no new formulas,
just running the existing logic across a finer grid and persisting it so
the dashboard doesn't recompute anything inline.

Outputs:
  data/corrected_cost_curve_baseline.csv   -- per-threshold: precision,
      recall, best pool_size, shortfall_prob, cost breakdown (baseline $)
  data/corrected_cost_curve_peak.csv       -- same, peak-season $
  data/dynamic_sim_daily_log_threshold_{1,30,50}.csv -- 180-day fleet
      health trajectory per reassignment-threshold policy
  data/catch_rate_comparison.csv           -- naive vs corrected proactive
      catch rate per threshold (the HDF/PWF chronic-loop fix)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from reassignment_sim_v2 import run_simulation, SIM_PARAMS

scored = pd.read_csv("data/scored_dataset.csv")

COST_PARAMS = {
    "repair_time_hours": 2.5,
    "downtime_cost_per_hour_baseline": 100_000,
    "downtime_cost_per_hour_peak": 260_000,
    "period_hours": 24,
    "inspection_pull_cost": 500,
    "inspection_downtime_hours": 0.5,
    "holding_cost_per_spare_per_period": 40,
    "cheap_swap_cost": 1500,
    # Unsourced placeholder, previously missing entirely (chronic reassignments
    # cost $0 in every prior version of this model). A task/environment
    # reassignment involves no physical part replacement, so it should cost
    # less than a repair-with-spare-available (cheap_swap_cost, $1,500), but
    # it requires reconfiguring the robot's task/routing assignment and
    # confirming the new profile actually resolves the issue, which is more
    # than a quick inspection pull ($500). Set at the midpoint of that range,
    # flagged as an assumption needing real calibration, same as swap_cost
    # and holding_cost were before they got grounded.
    "reassignment_cost": 800,
}
SIM_DAYS = SIM_PARAMS["sim_days"]
POOL_SIZES = np.array(range(0, 301, 1))
THRESHOLD_GRID = list(range(1, 100, 1))

print("Running corrected dynamic simulation across full threshold grid (1-99)...")
dyn_rows = []
catch_rows = []
daily_logs_to_save = {1: None, 30: None, 50: None}

for t in THRESHOLD_GRID:
    res = run_simulation(t)

    pred_positive = scored["health_score"] < t
    tp = int((pred_positive & (scored["failure"] == 1)).sum())
    fp = int((pred_positive & (scored["failure"] == 0)).sum())
    fn = int((~pred_positive & (scored["failure"] == 1)).sum())
    tn = int((~pred_positive & (scored["failure"] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    dyn_rows.append({
        "threshold": t, "precision": precision, "recall": recall,
        "static_TP": tp, "static_FP": fp, "static_FN": fn, "static_TN": tn,
        "genuine_catches_180d": res["genuine_catches"],
        "chronic_reassignments_180d": res["chronic_reassignments"],
        "unplanned_180d": res["unplanned_failures"],
    })

    old_rate = res["naive_proactive_catches"] / (res["naive_proactive_catches"] + res["unplanned_failures"]) if (res["naive_proactive_catches"] + res["unplanned_failures"]) > 0 else np.nan
    new_rate = res["genuine_catches"] / (res["genuine_catches"] + res["unplanned_failures"]) if (res["genuine_catches"] + res["unplanned_failures"]) > 0 else np.nan
    catch_rows.append({
        "threshold": t,
        "naive_proactive_catches": res["naive_proactive_catches"],
        "genuine_catches": res["genuine_catches"],
        "chronic_reassignments": res["chronic_reassignments"],
        "unplanned_failures": res["unplanned_failures"],
        "old_catch_rate": old_rate,
        "new_catch_rate": new_rate,
    })

    if t in daily_logs_to_save:
        daily_logs_to_save[t] = res["daily_log"]

    if t % 10 == 0:
        print(f"  ...threshold {t} done")

dyn = pd.DataFrame(dyn_rows)
catch_df = pd.DataFrame(catch_rows)

catch_df.to_csv("data/catch_rate_comparison.csv", index=False)
print("saved -> data/catch_rate_comparison.csv")

for t, log in daily_logs_to_save.items():
    log.to_csv(f"data/dynamic_sim_daily_log_threshold_{t}.csv", index=False)
    print(f"saved -> data/dynamic_sim_daily_log_threshold_{t}.csv")


def build_cost_curve(downtime_cost_per_hour, p=COST_PARAMS):
    full_fn_cost = p["repair_time_hours"] * downtime_cost_per_hour
    rows = []
    for _, row in dyn.iterrows():
        unplanned_180d = row["unplanned_180d"]
        repair_load_180d = row["genuine_catches_180d"] + row["chronic_reassignments_180d"] + row["unplanned_180d"]
        fp_1day = row["static_FP"]

        avg_down = (repair_load_180d / SIM_DAYS) * p["repair_time_hours"] / 24 \
            + fp_1day * p["inspection_downtime_hours"] / 24

        shortfall_prob_arr = 1 - poisson.cdf(POOL_SIZES, avg_down) if avg_down > 0 else np.zeros_like(POOL_SIZES, dtype=float)
        expected_cost_per_fn_arr = shortfall_prob_arr * full_fn_cost + (1 - shortfall_prob_arr) * p["cheap_swap_cost"]
        fn_cost_180d_arr = unplanned_180d * expected_cost_per_fn_arr

        fp_cost_180d = fp_1day * SIM_DAYS * p["inspection_pull_cost"]
        pool_cost_180d_arr = POOL_SIZES * p["holding_cost_per_spare_per_period"] * SIM_DAYS
        chronic_cost_180d = row["chronic_reassignments_180d"] * p["reassignment_cost"]

        total_cost_arr = fp_cost_180d + fn_cost_180d_arr + pool_cost_180d_arr + chronic_cost_180d
        idx = int(np.argmin(total_cost_arr))

        rows.append({
            "threshold": row["threshold"],
            "precision": row["precision"], "recall": row["recall"],
            "avg_robots_down": round(avg_down, 2),
            "pool_size": int(POOL_SIZES[idx]),
            "shortfall_prob": round(float(shortfall_prob_arr[idx]), 6),
            "fp_cost_180d": round(fp_cost_180d),
            "fn_cost_180d": round(float(fn_cost_180d_arr[idx])),
            "pool_cost_180d": round(float(pool_cost_180d_arr[idx])),
            "chronic_cost_180d": round(float(chronic_cost_180d)),
            "total_cost_180d": round(float(total_cost_arr[idx])),
        })
    return pd.DataFrame(rows)


print("Building corrected cost curves (baseline + peak season)...")
baseline_curve = build_cost_curve(COST_PARAMS["downtime_cost_per_hour_baseline"])
peak_curve = build_cost_curve(COST_PARAMS["downtime_cost_per_hour_peak"])

baseline_curve.to_csv("data/corrected_cost_curve_baseline.csv", index=False)
peak_curve.to_csv("data/corrected_cost_curve_peak.csv", index=False)
print("saved -> data/corrected_cost_curve_baseline.csv")
print("saved -> data/corrected_cost_curve_peak.csv")

print("\nDone. Baseline optimum:")
best = baseline_curve.loc[baseline_curve["total_cost_180d"].idxmin()]
print(best)
