"""
Piece 4, v2 -- fixes the HDF/PWF repair-loop artifact from v1.

v1 problem: HDF/PWF depend only on a robot's fixed torque/temp/rpm task
profile, not wear. Wear-reset repair can't fix that, so 207 robots got
flagged and "repaired" every single day for 180 days without the underlying
condition ever resolving -- inflating the proactive-catch count with
37,260 repeat non-fixes (84.6% of v1's threshold=1 total).

Fix: track each robot's CONSECUTIVE flagged-day streak. If a robot is still
flagged after CHRONIC_THRESHOLD consecutive days despite being "repaired"
each day, that's diagnostic of a condition repair can't reach -- treat it as
one chronic-misassignment event, not N fresh catches, and resolve it with a
task/environment reassignment instead: draw a fresh (torque, air_temp,
process_temp, rpm) profile from the pool of currently-safe robots, exactly
modeling "give this robot different work" rather than "fix its part."

This mechanic is general (driven by the recurrence PATTERN, not a hardcoded
HDF/PWF check) -- it would catch any case where repair doesn't resolve the
underlying driver, not just the two modes we already know are wear-
independent.

Reports BOTH the naive catch rate (every flagged day counted, v1's method)
and the corrected rate (episodes counted once, chronic episodes excluded
entirely) side by side, plus the chronic-reassignment count as its own
bucket.
"""

import numpy as np
import pandas as pd

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
types = df["Type"].values
osf_thresh = np.array([OSF_THRESHOLD[t] for t in types], dtype=float)

SIM_PARAMS = {
    "sim_days": 180,
    "base_wear_rate_per_day": 1.0,
    "receiver_wear_multiplier": 1.5,
    "twf_daily_hazard_in_window": 0.005,
    "rnf_daily_hazard": 0.001,
    "chronic_threshold_days": 3,   # consecutive flagged days before we call it chronic misassignment
}


def compute_health(wear, torque, air_temp, process_temp, rpm, osf_thresh):
    ratio = (torque * wear) / osf_thresh
    r_osf = np.clip((ratio - 0.5) / 0.5, 0, 1)

    temp_diff = process_temp - air_temp
    margin_temp = temp_diff - 8.6
    margin_rpm = rpm - 1380
    unsafe_temp = np.clip(1.0 - margin_temp / 3.0, 0, 1)
    unsafe_rpm = np.clip(1.0 - margin_rpm / 150.0, 0, 1)
    r_hdf = unsafe_temp * unsafe_rpm

    power = torque * (rpm * 2 * np.pi / 60)
    headroom_low = power - 3500
    headroom_high = 9000 - power
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
    torque = df["torque"].values.astype(float).copy()
    air_temp = df["air_temp"].values.astype(float).copy()
    process_temp = df["process_temp"].values.astype(float).copy()
    rpm = df["rpm"].values.astype(float).copy()

    consecutive_flag_days = np.zeros(N, dtype=int)
    repair_count = np.zeros(N, dtype=int)
    receiver_count = np.zeros(N, dtype=int)

    naive_proactive_catches = 0      # v1 method: every flagged day counted
    genuine_catches = 0              # v2: episodes only, chronic ones excluded
    chronic_reassignments = 0
    unplanned_failures = 0
    unplanned_failure_modes = {"OSF": 0, "HDF": 0, "PWF": 0, "TWF": 0, "RNF": 0}
    daily_log = []

    for day in range(p["sim_days"]):
        health = compute_health(wear, torque, air_temp, process_temp, rpm, osf_thresh)
        flagged = health < reassignment_threshold

        actual_osf = (torque * wear) > osf_thresh
        temp_diff = process_temp - air_temp
        actual_hdf = (temp_diff < 8.6) & (rpm < 1380)
        power = torque * (rpm * 2 * np.pi / 60)
        actual_pwf = (power < 3500) | (power > 9000)
        in_window = (wear >= 200) & (wear <= 240)
        twf_roll = in_window & (rng.random(N) < p["twf_daily_hazard_in_window"])
        rnf_roll = rng.random(N) < p["rnf_daily_hazard"]
        actual_fail = actual_osf | actual_hdf | actual_pwf | twf_roll | rnf_roll

        unplanned = actual_fail & ~flagged
        n_unplanned = int(unplanned.sum())
        unplanned_failures += n_unplanned
        if n_unplanned > 0:
            unplanned_failure_modes["OSF"] += int((unplanned & actual_osf).sum())
            unplanned_failure_modes["HDF"] += int((unplanned & actual_hdf).sum())
            unplanned_failure_modes["PWF"] += int((unplanned & actual_pwf).sum())
            unplanned_failure_modes["TWF"] += int((unplanned & twf_roll).sum())
            unplanned_failure_modes["RNF"] += int((unplanned & rnf_roll).sum())

        naive_proactive_catches += int(flagged.sum())

        # episode tracking: a new episode starts the first day a robot is
        # flagged after not being flagged; provisionally counted as genuine
        episode_start_today = flagged & (consecutive_flag_days == 0)
        genuine_catches += int(episode_start_today.sum())

        new_consecutive = np.where(flagged, consecutive_flag_days + 1, 0)
        became_chronic_today = flagged & (new_consecutive == p["chronic_threshold_days"])

        # reclassify: undo the provisional genuine-catch count for episodes
        # that just crossed into chronic territory
        chronic_reassignments += int(became_chronic_today.sum())
        genuine_catches -= int(became_chronic_today.sum())

        consecutive_flag_days = new_consecutive
        consecutive_flag_days[became_chronic_today] = 0  # resolved today, streak clears

        being_repaired = flagged | unplanned
        repair_count[being_repaired] += 1

        # task/environment reassignment for chronic robots: draw a fresh
        # profile from the currently-safe population
        if became_chronic_today.any():
            safe_mask = ~actual_hdf & ~actual_pwf
            safe_idx = np.where(safe_mask)[0]
            chronic_idx = np.where(became_chronic_today)[0]
            donors = rng.choice(safe_idx, size=len(chronic_idx), replace=True)
            torque[chronic_idx] = torque[donors]
            air_temp[chronic_idx] = air_temp[donors]
            process_temp[chronic_idx] = process_temp[donors]
            rpm[chronic_idx] = rpm[donors]

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

        multiplier = np.ones(N)
        multiplier[receivers_idx] = p["receiver_wear_multiplier"]
        wear_next = wear + p["base_wear_rate_per_day"] * multiplier
        wear_next[being_repaired] = 0.0
        wear = wear_next

        daily_log.append({
            "day": day, "flagged": int(flagged.sum()), "unplanned": n_unplanned,
            "chronic_today": int(became_chronic_today.sum()),
            "mean_health": round(float(health.mean()), 2),
            "pct_fleet_health_lt_50": round(float((health < 50).mean() * 100), 2),
        })

    return {
        "naive_proactive_catches": naive_proactive_catches,
        "genuine_catches": genuine_catches,
        "chronic_reassignments": chronic_reassignments,
        "unplanned_failures": unplanned_failures,
        "unplanned_failure_modes": unplanned_failure_modes,
        "receiver_count": receiver_count,
        "daily_log": pd.DataFrame(daily_log),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    for threshold in [1, 30, 50]:
        print("=" * 70)
        print(f"REASSIGNMENT THRESHOLD = {threshold}")
        print("=" * 70)
        result = run_simulation(threshold)

        old_rate = result["naive_proactive_catches"] / (result["naive_proactive_catches"] + result["unplanned_failures"])
        new_rate = result["genuine_catches"] / (result["genuine_catches"] + result["unplanned_failures"])

        print(f"Over {SIM_PARAMS['sim_days']} days, fleet of {N}:")
        print(f"  naive proactive catches (v1 method, every flagged day):  {result['naive_proactive_catches']}")
        print(f"  genuine proactive catches (v2, episodes, chronic excl.): {result['genuine_catches']}")
        print(f"  chronic misassignment events (task/env reassignment):    {result['chronic_reassignments']}")
        print(f"  unplanned failures:                                      {result['unplanned_failures']}")
        print(f"  OLD catch rate (v1, inflated by repeat non-fixes): {old_rate:.1%}")
        print(f"  NEW catch rate (v2, genuine wear-driven catches):  {new_rate:.1%}")
        print(f"  unplanned failure breakdown by mode: {result['unplanned_failure_modes']}")

        dl = result["daily_log"]
        print(f"\n  fleet mean health: day 0 -> {dl.iloc[0]['mean_health']}, day {SIM_PARAMS['sim_days']-1} -> {dl.iloc[-1]['mean_health']}")
        print(f"  fleet % below health 50: day 0 -> {dl.iloc[0]['pct_fleet_health_lt_50']}%, "
              f"day {SIM_PARAMS['sim_days']-1} -> {dl.iloc[-1]['pct_fleet_health_lt_50']}%")
        print()
