"""
Shortfall-gated cost model + joint (threshold, pool_size) optimization.

Correctness fix over the previous version: FN cost is no longer a flat
repair_time_hours * downtime_cost_per_hour applied to every miss (which
implicitly assumed no backup pool exists). Now:

    shortfall_prob(pool_size, avg_down) = P(concurrent_down > pool_size)
                                         = 1 - Poisson.cdf(pool_size, avg_down)

    expected_cost_per_FN = shortfall_prob * full_downtime_cost
                          + (1 - shortfall_prob) * cheap_swap_cost

    full_downtime_cost   = repair_time_hours * downtime_cost_per_hour
                            (the pool has no spare -- the station genuinely
                            stops for the full repair duration)
    cheap_swap_cost       = a spare is available -- swap it in immediately,
                            repair the failed unit off-line, no line stoppage
                            (NEW assumption parameter, not previously cited)

FP cost is left as the flat, non-benchmark-linked inspection_pull_cost from
the prior model -- the shortfall gate specifically targets FN cost, per this
turn's instruction (a false-alarm pull isn't really a fleet-availability
event the way an unplanned failure is).

Structural change: threshold and pool_size are now interdependent (a given
threshold's concurrent-down load determines how often a given pool size gets
exhausted, which determines expected FN cost, which feeds back into which
threshold is cheapest). So for every threshold, we sweep the FULL pool_size
range and take that threshold's best pool size, then compare across
thresholds -- a true joint optimization, not sequential threshold-then-pool.

The 95%-service-level target from the previous model is dropped here: pool
size is no longer sized to hit an arbitrary service level, it's chosen
purely to minimize total cost, exactly like the threshold is.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

df = pd.read_csv("data/scored_dataset.csv")
FLEET_SIZE = len(df)

PARAMS = {
    # --- cited benchmarks (adjustable) ---
    "repair_time_hours": 2.5,                    # midpoint of cited 1-4 hr range
    "downtime_cost_per_hour_baseline": 100_000,  # low end of cited $100K-$260K/hr range
    "downtime_cost_per_hour_peak": 260_000,       # high end of the SAME cited range

    # --- structural assumption ---
    "period_hours": 24,

    # --- flagged assumptions, adjustable ---
    "inspection_pull_cost": 500,        # flat cost of an unnecessary pull/check
    "inspection_downtime_hours": 0.5,   # used only for pool-capacity load, not directly monetized
    "holding_cost_per_spare_per_period": 100,
    "cheap_swap_cost": 1500,            # NEW: cost of an FN repair when a spare IS available -- swap + off-line repair, no line stoppage
}

print("=" * 70)
print("TRANSPARENCY CHECK -- exact values behind last run's per-FN-event cost")
print("=" * 70)
print(f"repair_time_hours = {PARAMS['repair_time_hours']}")
print(f"downtime_cost_per_hour (baseline) = ${PARAMS['downtime_cost_per_hour_baseline']:,}")
print(f"  -> full per-FN-event cost (baseline) = {PARAMS['repair_time_hours']} x ${PARAMS['downtime_cost_per_hour_baseline']:,} = ${PARAMS['repair_time_hours']*PARAMS['downtime_cost_per_hour_baseline']:,.0f}")
print(f"downtime_cost_per_hour (peak) = ${PARAMS['downtime_cost_per_hour_peak']:,}")
print(f"  -> full per-FN-event cost (peak) = {PARAMS['repair_time_hours']} x ${PARAMS['downtime_cost_per_hour_peak']:,} = ${PARAMS['repair_time_hours']*PARAMS['downtime_cost_per_hour_peak']:,.0f}")
print("(these are the 'full downtime cost' figures used ONLY when the pool is exhausted, below)")
print()


def threshold_flow(threshold):
    pred_positive = df["health_score"] < threshold
    tp = int((pred_positive & (df["failure"] == 1)).sum())
    fp = int((pred_positive & (df["failure"] == 0)).sum())
    fn = int((~pred_positive & (df["failure"] == 1)).sum())
    tn = int((~pred_positive & (df["failure"] == 0)).sum())
    return tp, fp, fn, tn


def joint_sweep(downtime_cost_per_hour, thresholds, pool_sizes, p=PARAMS):
    full_fn_cost = p["repair_time_hours"] * downtime_cost_per_hour
    pool_arr = np.array(pool_sizes)
    rows = []
    for t in thresholds:
        tp, fp, fn, tn = threshold_flow(t)
        avg_down = ((tp + fn) * p["repair_time_hours"] + fp * p["inspection_downtime_hours"]) / p["period_hours"]

        if avg_down > 0:
            shortfall_prob_arr = 1 - poisson.cdf(pool_arr, avg_down)
        else:
            shortfall_prob_arr = np.zeros_like(pool_arr, dtype=float)

        expected_cost_per_fn_arr = shortfall_prob_arr * full_fn_cost + (1 - shortfall_prob_arr) * p["cheap_swap_cost"]
        fn_cost_arr = fn * expected_cost_per_fn_arr
        fp_cost_total = fp * p["inspection_pull_cost"]
        pool_cost_arr = pool_arr * p["holding_cost_per_spare_per_period"]
        total_cost_arr = fp_cost_total + fn_cost_arr + pool_cost_arr

        best_idx = int(np.argmin(total_cost_arr))
        rows.append({
            "threshold": t,
            "TP": tp, "FP": fp, "FN": fn,
            "avg_down": round(avg_down, 1),
            "best_pool_size": pool_sizes[best_idx],
            "shortfall_prob_at_best_pool": round(shortfall_prob_arr[best_idx], 4),
            "fp_cost": round(fp_cost_total),
            "fn_cost_at_best_pool": round(fn_cost_arr[best_idx]),
            "pool_cost_at_best_pool": round(pool_cost_arr[best_idx]),
            "total_cost": round(total_cost_arr[best_idx]),
        })
    curve = pd.DataFrame(rows)
    return curve


thresholds = list(range(1, 100, 1))
pool_sizes = list(range(0, 251, 1))

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

scenarios = {
    "BASELINE": PARAMS["downtime_cost_per_hour_baseline"],
    "PEAK_SEASON": PARAMS["downtime_cost_per_hour_peak"],
}

results = {}
for label, dtc in scenarios.items():
    curve = joint_sweep(dtc, thresholds, pool_sizes)
    results[label] = curve
    best = curve.loc[curve["total_cost"].idxmin()]

    print("=" * 70)
    print(f"{label}  (downtime_cost_per_hour = ${dtc:,})")
    print("=" * 70)
    display_rows = curve[curve["threshold"] % 5 == 0].copy()
    best_row_df = curve.loc[[curve["total_cost"].idxmin()]]
    display_rows = pd.concat([display_rows, best_row_df]).drop_duplicates(subset="threshold").sort_values("threshold")
    print(display_rows.to_string(index=False))
    print(f"\n>>> JOINT OPTIMUM: threshold={best['threshold']:.0f}, pool_size={best['best_pool_size']:.0f}  "
          f"(total_cost=${best['total_cost']:,.0f}, avg_down={best['avg_down']:.1f}, "
          f"shortfall_prob={best['shortfall_prob_at_best_pool']:.2%}, "
          f"FN={best['FN']:.0f}, FP={best['FP']:.0f})")
    print()
    curve.to_csv(f"data/shortfall_gated_curve_{label.lower()}.csv", index=False)

print("=" * 70)
print("SEASONAL SHIFT SUMMARY (shortfall-gated, joint threshold+pool optimization)")
print("=" * 70)
b = results["BASELINE"].loc[results["BASELINE"]["total_cost"].idxmin()]
pk = results["PEAK_SEASON"].loc[results["PEAK_SEASON"]["total_cost"].idxmin()]
print(f"Baseline:    threshold={b['threshold']:.0f}, pool_size={b['best_pool_size']:.0f}, total_cost=${b['total_cost']:,.0f}")
print(f"Peak season: threshold={pk['threshold']:.0f}, pool_size={pk['best_pool_size']:.0f}, total_cost=${pk['total_cost']:,.0f}")
print(f"Shift: {pk['threshold']-b['threshold']:+.0f} threshold points, {pk['best_pool_size']-b['best_pool_size']:+.0f} pool slots")
