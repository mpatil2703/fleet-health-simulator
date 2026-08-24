"""
Fixes the static cost model's missing term: it evaluated FN cost from a
ONE-SHOT snapshot (a single day's flagged/missed counts, implicitly assumed
constant forever). The dynamic simulation showed that assumption breaks at
low thresholds -- degradation compounds, so the true cumulative unplanned-
failure count over a real time horizon is NOT just (one-day rate x days).

Fix: run the corrected (v2, chronic-loop-fixed) 180-day dynamic simulation
across a threshold grid, and use its ACTUAL cumulative unplanned-failure
count per threshold as the FN term, instead of extrapolating a single day's
snapshot. FP cost is left on the static one-shot approximation (scaled to
the 180-day horizon) -- there's no clean dynamic counterfactual for "would
this flagged robot actually have failed," so this piece is NOT corrected
here; flagged explicitly as a remaining simplification, and its small
relative weight (established earlier: FN cost dominates FP cost by ~500x
per event) means it's unlikely to be biasing the comparison materially.

Then re-solve the joint (threshold, pool_size) optimum with this corrected
cost, for baseline and peak-season downtime cost, and compare to the
earlier (uncorrected, one-shot) answer: threshold=1, pool_size=64-65.
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
    "holding_cost_per_spare_per_period": 40,   # grounded midpoint of corrected $15-65/day range
    "cheap_swap_cost": 1500,                    # grounded per prior turn
}

SIM_DAYS = SIM_PARAMS["sim_days"]
THRESHOLD_GRID = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
POOL_SIZES = np.array(range(0, 301, 1))


def fp_static_count(threshold):
    pred_positive = scored["health_score"] < threshold
    return int((pred_positive & (scored["failure"] == 0)).sum())


print("=" * 70)
print("STEP 1 -- run corrected (v2) dynamic simulation across threshold grid")
print("=" * 70)

dynamic_rows = []
for t in THRESHOLD_GRID:
    res = run_simulation(t)
    fp_static = fp_static_count(t)
    dynamic_rows.append({
        "threshold": t,
        "genuine_catches_180d": res["genuine_catches"],
        "chronic_reassignments_180d": res["chronic_reassignments"],
        "unplanned_180d": res["unplanned_failures"],
        "fp_static_1day": fp_static,
    })
    print(f"  threshold={t:3d}  genuine={res['genuine_catches']:6d}  chronic={res['chronic_reassignments']:5d}  "
          f"unplanned_180d={res['unplanned_failures']:5d}  fp_static/day={fp_static:5d}")

dyn = pd.DataFrame(dynamic_rows)

# static one-shot FN estimate scaled naively to 180 days, for direct comparison
static_fn_1day = {}
for t in THRESHOLD_GRID:
    pred_positive = scored["health_score"] < t
    fn = int((~pred_positive & (scored["failure"] == 1)).sum())
    static_fn_1day[t] = fn
dyn["static_fn_1day"] = dyn["threshold"].map(static_fn_1day)
dyn["static_fn_180d_naive_estimate"] = dyn["static_fn_1day"] * SIM_DAYS
dyn["compounding_gap"] = dyn["unplanned_180d"] - dyn["static_fn_180d_naive_estimate"]
dyn["compounding_gap_pct"] = (dyn["compounding_gap"] / dyn["static_fn_180d_naive_estimate"] * 100).round(1)

print("\n--- compounding check: dynamic cumulative unplanned vs naive static x180 estimate ---")
print(dyn[["threshold", "static_fn_1day", "static_fn_180d_naive_estimate", "unplanned_180d", "compounding_gap", "compounding_gap_pct"]].to_string(index=False))

dyn.to_csv("data/dynamic_threshold_grid.csv", index=False)


def joint_optimum_corrected(downtime_cost_per_hour, p=COST_PARAMS):
    full_fn_cost = p["repair_time_hours"] * downtime_cost_per_hour
    best = None
    rows = []
    for _, row in dyn.iterrows():
        t = row["threshold"]
        unplanned_180d = row["unplanned_180d"]
        repair_load_180d = row["genuine_catches_180d"] + row["chronic_reassignments_180d"] + row["unplanned_180d"]
        fp_1day = row["fp_static_1day"]

        avg_down = (repair_load_180d / SIM_DAYS) * p["repair_time_hours"] / 24 \
            + fp_1day * p["inspection_downtime_hours"] / 24

        shortfall_prob_arr = 1 - poisson.cdf(POOL_SIZES, avg_down) if avg_down > 0 else np.zeros_like(POOL_SIZES, dtype=float)
        expected_cost_per_fn_arr = shortfall_prob_arr * full_fn_cost + (1 - shortfall_prob_arr) * p["cheap_swap_cost"]
        fn_cost_180d_arr = unplanned_180d * expected_cost_per_fn_arr

        fp_cost_180d = fp_1day * SIM_DAYS * p["inspection_pull_cost"]
        pool_cost_180d_arr = POOL_SIZES * p["holding_cost_per_spare_per_period"] * SIM_DAYS

        total_cost_arr = fp_cost_180d + fn_cost_180d_arr + pool_cost_180d_arr
        idx = int(np.argmin(total_cost_arr))

        record = {
            "threshold": t, "avg_down": round(avg_down, 1), "pool_size": POOL_SIZES[idx],
            "shortfall_prob": round(shortfall_prob_arr[idx], 5),
            "fp_cost_180d": round(fp_cost_180d), "fn_cost_180d": round(fn_cost_180d_arr[idx]),
            "pool_cost_180d": round(pool_cost_180d_arr[idx]), "total_cost_180d": round(total_cost_arr[idx]),
        }
        rows.append(record)
        if best is None or record["total_cost_180d"] < best["total_cost_180d"]:
            best = record
    return pd.DataFrame(rows), best


pd.set_option("display.width", 160)

print("\n" + "=" * 70)
print("STEP 2 -- re-solve joint (threshold, pool_size) with corrected 180-day cost")
print("=" * 70)

scenarios = {
    "BASELINE": COST_PARAMS["downtime_cost_per_hour_baseline"],
    "PEAK_SEASON": COST_PARAMS["downtime_cost_per_hour_peak"],
}

bests = {}
for label, dtc in scenarios.items():
    curve, best = joint_optimum_corrected(dtc)
    bests[label] = best
    print(f"\n{label}  (downtime_cost_per_hour = ${dtc:,})")
    print(curve.to_string(index=False))
    print(f">>> CORRECTED JOINT OPTIMUM: threshold={best['threshold']}, pool_size={best['pool_size']}, "
          f"total_cost_180d=${best['total_cost_180d']:,}")

print("\n" + "=" * 70)
print("COMPARISON: previous (one-shot) vs corrected (180-day cumulative) optimum")
print("=" * 70)
print("Previous (one-shot snapshot, shortfall-gated): threshold=1, pool_size=64-65 (per-day cost ~$100K)")
for label, best in bests.items():
    print(f"Corrected ({label}): threshold={best['threshold']}, pool_size={best['pool_size']}, "
          f"total_cost over 180 days=${best['total_cost_180d']:,} "
          f"(~${best['total_cost_180d']/SIM_DAYS:,.0f}/day equivalent)")
