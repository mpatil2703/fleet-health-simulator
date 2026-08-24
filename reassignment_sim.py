"""
Piece 4 -- threshold-based task reassignment with a wear-acceleration
feedback loop.

AI4I is cross-sectional, so this is a forward-in-time simulation seeded
from the real 10,000-row snapshot as t=0 state. Per-robot Type/torque/
air_temp/process_temp/rpm are held FIXED (they represent a robot's task/
environment profile, not something that changes day to day); only `wear`
evolves, exactly mirroring how it's used everywhere else in this project
(the validated usage-intensity proxy).

Design (confirmed):
  - Receiver policy: N healthiest fleet-wide, excluding robots that are
    themselves flagged/repairing or already a receiver that day.
  - Wear mechanic: flat multiplier (1.5x) on a receiver's wear-accumulation
    rate for each day it's carrying reassigned load.
  - A flagged robot is pulled for the WHOLE day it's flagged (a
    simplification of the 1-4 hr repair_time_hours benchmark into a
    day-stepped model -- real wrench time is short, but scheduling/logistics
    overhead removes it from productive service for the day), and returns
    the next day with wear reset to 0 (part/tool replaced).
  - Ground truth "actual failure" uses the SAME validated deterministic
    rules as the rest of this project: OSF = torque*wear > Type threshold,
    HDF/PWF = the published fixed-condition rules (static per robot, since
    they depend only on the fixed temp/rpm/torque profile, not wear), TWF =
    stochastic hazard while wear sits in [200,240] (calibrated, flagged
    assumption), RNF = flat daily hazard (published baseline rate).
  - If an actual failure fires on a robot the health-score policy did NOT
    flag that day, that's an unplanned/missed event -- it still forces a
    repair (wear resets), it's just discovered reactively instead of
    proactively.

Flagged assumptions in this piece (adjustable, not previously specified):
  base_wear_rate_per_day, receiver_wear_multiplier (=1.5, per confirmed
  design), twf_daily_hazard_in_window, rnf_daily_hazard, sim_days.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

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

N = len(df)
OSF_THRESHOLD = {"L": 11000, "M": 12000, "H": 13000}

SIM_PARAMS = {
    "sim_days": 180,
    "base_wear_rate_per_day": 1.0,          # assumption: full life (~0->250) at baseline pace ~250 days
    "receiver_wear_multiplier": 1.5,        # confirmed design choice
    "twf_daily_hazard_in_window": 0.005,    # assumption, calibrated loosely to AI4I's observed in-window hit rate
    "rnf_daily_hazard": 0.001,              # matches published RNF generating rate
}

# fixed per-robot attributes
torque0 = df["torque"].values.astype(float)
air_temp0 = df["air_temp"].values.astype(float)
process_temp0 = df["process_temp"].values.astype(float)
rpm0 = df["rpm"].values.astype(float)
types = df["Type"].values
osf_thresh = np.array([OSF_THRESHOLD[t] for t in types], dtype=float)

# static (wear-independent) actual-failure conditions, precomputed once
temp_diff0 = process_temp0 - air_temp0
actual_hdf_static = (temp_diff0 < 8.6) & (rpm0 < 1380)
power0 = torque0 * (rpm0 * 2 * np.pi / 60)
actual_pwf_static = (power0 < 3500) | (power0 > 9000)


def compute_health(wear, p=SIM_PARAMS):
    ratio = (torque0 * wear) / osf_thresh
    r_osf = np.clip((ratio - 0.5) / 0.5, 0, 1)

    margin_temp = temp_diff0 - 8.6
    margin_rpm = rpm0 - 1380
    unsafe_temp = np.clip(1.0 - margin_temp / 3.0, 0, 1)
    unsafe_rpm = np.clip(1.0 - margin_rpm / 150.0, 0, 1)
    r_hdf = unsafe_temp * unsafe_rpm

    headroom_low = power0 - 3500
    headroom_high = 9000 - power0
    margin = np.minimum(headroom_low, headroom_high)
    r_pwf = np.clip(1.0 - margin / 500.0, 0, 1)

    in_window = (wear >= 180) & (wear <= 240)
    frac = np.clip((wear - 180) / 60.0, 0, 1)
    r_twf = np.where(in_window, frac * 0.40, 0.0)

    r_rnf = 0.001
    p_fail = 1 - (1 - r_osf) * (1 - r_hdf) * (1 - r_pwf) * (1 - r_twf) * (1 - r_rnf)
    return 100 * (1 - p_fail)


def run_simulation(reassignment_threshold, p=SIM_PARAMS, seed=42):
    rng = np.random.default_rng(seed)
    wear = df["wear"].values.astype(float).copy()

    repair_count = np.zeros(N, dtype=int)
    receiver_count = np.zeros(N, dtype=int)
    proactive_catches = 0
    unplanned_failures = 0
    unplanned_failure_modes = {"OSF": 0, "HDF": 0, "PWF": 0, "TWF": 0, "RNF": 0}
    daily_log = []

    for day in range(p["sim_days"]):
        health = compute_health(wear, p)
        flagged = health < reassignment_threshold

        # actual mechanistic failure check on TODAY's (unrepaired) wear
        actual_osf = (torque0 * wear) > osf_thresh
        in_window = (wear >= 200) & (wear <= 240)
        twf_roll = in_window & (rng.random(N) < p["twf_daily_hazard_in_window"])
        rnf_roll = rng.random(N) < p["rnf_daily_hazard"]
        actual_fail = actual_osf | actual_hdf_static | actual_pwf_static | twf_roll | rnf_roll

        unplanned = actual_fail & ~flagged
        proactive = flagged

        n_unplanned = int(unplanned.sum())
        n_proactive = int(proactive.sum())
        unplanned_failures += n_unplanned
        proactive_catches += n_proactive
        if n_unplanned > 0:
            unplanned_failure_modes["OSF"] += int((unplanned & actual_osf).sum())
            unplanned_failure_modes["HDF"] += int((unplanned & actual_hdf_static).sum())
            unplanned_failure_modes["PWF"] += int((unplanned & actual_pwf_static).sum())
            unplanned_failure_modes["TWF"] += int((unplanned & twf_roll).sum())
            unplanned_failure_modes["RNF"] += int((unplanned & rnf_roll).sum())

        being_repaired = flagged | unplanned
        repair_count[being_repaired] += 1

        # receiver selection: N-healthiest among robots NOT being repaired today
        n_need_coverage = int(being_repaired.sum())
        eligible_idx = np.where(~being_repaired)[0]
        receivers_idx = np.array([], dtype=int)
        if n_need_coverage > 0 and len(eligible_idx) > 0:
            n_receivers = min(n_need_coverage, len(eligible_idx))
            eligible_health = health[eligible_idx]
            top_order = np.argsort(-eligible_health)[:n_receivers]
            receivers_idx = eligible_idx[top_order]
            receiver_count[receivers_idx] += 1

        # update wear
        multiplier = np.ones(N)
        multiplier[receivers_idx] = p["receiver_wear_multiplier"]
        wear_next = wear + p["base_wear_rate_per_day"] * multiplier
        wear_next[being_repaired] = 0.0  # repaired today -> fresh part, resets for tomorrow
        wear = wear_next

        daily_log.append({
            "day": day, "flagged": int(flagged.sum()), "unplanned": n_unplanned,
            "receivers": len(receivers_idx),
            "mean_health": round(float(health.mean()), 2),
            "pct_fleet_health_lt_50": round(float((health < 50).mean() * 100), 2),
        })

    return {
        "proactive_catches": proactive_catches,
        "unplanned_failures": unplanned_failures,
        "unplanned_failure_modes": unplanned_failure_modes,
        "repair_count": repair_count,
        "receiver_count": receiver_count,
        "daily_log": pd.DataFrame(daily_log),
    }


pd.set_option("display.width", 160)

for threshold in [1, 30, 50]:
    print("=" * 70)
    print(f"REASSIGNMENT THRESHOLD = {threshold}")
    print("=" * 70)
    result = run_simulation(threshold)

    total_events = result["proactive_catches"] + result["unplanned_failures"]
    catch_rate = result["proactive_catches"] / total_events if total_events > 0 else float("nan")
    print(f"Over {SIM_PARAMS['sim_days']} simulated days, fleet of {N}:")
    print(f"  proactive catches (flagged before failing): {result['proactive_catches']}")
    print(f"  unplanned failures (missed by policy):       {result['unplanned_failures']}")
    print(f"  proactive catch rate:                        {catch_rate:.1%}")
    print(f"  unplanned failure breakdown by mode: {result['unplanned_failure_modes']}")

    rc = result["receiver_count"]
    print(f"\n  receiver load distribution (times selected as a load-absorbing receiver):")
    print(f"    robots never used as receiver: {(rc == 0).sum()} / {N}")
    print(f"    max times any single robot was picked as receiver: {rc.max()}")
    print(f"    mean among robots used at least once: {rc[rc > 0].mean():.2f}" if (rc > 0).any() else "    (none used)")
    top_receivers = np.argsort(-rc)[:5]
    print(f"    top 5 most-reused receivers (robot index, times used, Type, final wear):")
    for idx in top_receivers:
        print(f"      idx={idx}, used={rc[idx]}x, Type={types[idx]}, repairs_triggered={result['repair_count'][idx]}")

    dl = result["daily_log"]
    print(f"\n  fleet mean health: day 0 -> {dl.iloc[0]['mean_health']}, day {SIM_PARAMS['sim_days']-1} -> {dl.iloc[-1]['mean_health']}")
    print(f"  fleet % below health 50: day 0 -> {dl.iloc[0]['pct_fleet_health_lt_50']}%, "
          f"day {SIM_PARAMS['sim_days']-1} -> {dl.iloc[-1]['pct_fleet_health_lt_50']}%")
    print()
