"""
Fleet health-score model — v1.

Design (per validated findings in cohort_check.py / validate_thresholds.py):
  - OSF (overstrain) is the PRIMARY signal. It is a deterministic, Type-adjusted
    threshold in the real data generating process (torque x wear > {L:11000,
    M:12000, H:13000} minNm, confirmed 100% match against all 10,000 rows).
    Cohort finding: L has a genuinely lower tolerance than M/H, which are
    statistically indistinguishable from each other (not a 3-tier gradient).
  - HDF and PWF are SECONDARY signals, each with their own deterministic
    mechanism (also confirmed 100% match against published rules):
        HDF: (process_temp - air_temp) < 8.6 K  AND  rpm < 1380
        PWF: power = torque * rpm[rad/s]  outside [3500, 9000] W
  - TWF is a SECONDARY signal but epistemically different: each unit has a
    hidden, per-unit-random wear-out point in [200,240] min tool wear, and
    only ~5.4% of rows sitting in that window actually trip TWF in this
    dataset. Sensor data cannot resolve which unit is "the one" -- so TWF
    risk is deliberately capped low/moderate, not driven to certainty.
  - RNF gets a flat baseline risk term only (published generating rate
    0.1% per row, independent of all sensor parameters) -- explicitly NOT
    a predictive signal, since near-miss analysis confirmed it carries no
    sensor precursor.

All thresholds/ramp widths below are named parameters, not magic numbers --
this is meant to be an adjustable framework, not a fixed fit.
"""

import numpy as np
import pandas as pd

PARAMS = {
    # OSF -- primary. Ramp starts at this fraction of the Type threshold
    # and reaches "critical" exactly at the deterministic threshold (ratio=1).
    "osf_threshold": {"L": 11000, "M": 12000, "H": 13000},
    "osf_ramp_start_ratio": 0.5,   # risk = 0 at 50% of threshold

    # HDF -- secondary. Soft-AND of two margins; each ramps over this many
    # units of headroom above its hard threshold before risk starts.
    "hdf_temp_ref_range_k": 3.0,     # K of headroom over the 8.6K cutoff
    "hdf_rpm_ref_range": 150,        # rpm of headroom over the 1380 cutoff

    # PWF -- secondary. Risk ramps over this many watts of headroom to the
    # nearest of the two power bounds [3500, 9000] W.
    "pwf_ref_range_w": 500,

    # TWF -- secondary, capped. Ramp window on tool wear, and a max risk
    # ceiling reflecting the ~5-15% real-world hit rate in this band, not 100%.
    "twf_ramp_start_min": 180,
    "twf_ramp_end_min": 240,
    "twf_max_risk": 0.40,

    # RNF -- flat baseline only, published generating probability.
    "rnf_baseline": 0.001,
}


def _clip01(x):
    return max(0.0, min(1.0, x))


def osf_risk(torque, wear, product_type, p=PARAMS):
    threshold = p["osf_threshold"][product_type]
    ratio = (torque * wear) / threshold
    ramp_start = p["osf_ramp_start_ratio"]
    if ratio <= ramp_start:
        return 0.0, ratio
    if ratio >= 1.0:
        return 1.0, ratio
    return _clip01((ratio - ramp_start) / (1.0 - ramp_start)), ratio


def hdf_risk(air_temp, process_temp, rpm, p=PARAMS):
    temp_diff = process_temp - air_temp
    margin_temp = temp_diff - 8.6
    margin_rpm = rpm - 1380
    unsafe_temp = _clip01(1.0 - margin_temp / p["hdf_temp_ref_range_k"])
    unsafe_rpm = _clip01(1.0 - margin_rpm / p["hdf_rpm_ref_range"])
    return unsafe_temp * unsafe_rpm, (margin_temp, margin_rpm)


def pwf_risk(torque, rpm, p=PARAMS):
    power_w = torque * (rpm * 2 * np.pi / 60)
    headroom_low = power_w - 3500
    headroom_high = 9000 - power_w
    margin = min(headroom_low, headroom_high)
    risk = _clip01(1.0 - margin / p["pwf_ref_range_w"])
    return risk, (power_w, margin)


def twf_risk(wear, p=PARAMS):
    start, end, cap = p["twf_ramp_start_min"], p["twf_ramp_end_min"], p["twf_max_risk"]
    if wear < start:
        return 0.0
    if wear > end:
        # past the whole possible wear-out window: if it hasn't happened,
        # this unit's hidden trigger point already resolved (replaced) --
        # residual risk decays back down rather than staying elevated.
        return 0.0
    frac = (wear - start) / (end - start)
    return _clip01(frac) * cap


def compute_health(row, p=PARAMS):
    r_osf, osf_ratio = osf_risk(row["torque"], row["wear"], row["Type"], p)
    r_hdf, hdf_margins = hdf_risk(row["air_temp"], row["process_temp"], row["rpm"], p)
    r_pwf, pwf_info = pwf_risk(row["torque"], row["rpm"], p)
    r_twf = twf_risk(row["wear"], p)
    r_rnf = p["rnf_baseline"]

    risks = {"OSF": r_osf, "HDF": r_hdf, "PWF": r_pwf, "TWF": r_twf, "RNF": r_rnf}
    p_fail = 1.0 - np.prod([1.0 - v for v in risks.values()])
    health = 100.0 * (1.0 - p_fail)

    # Normalize each mode against its OWN max-possible risk before ranking --
    # otherwise TWF (capped at 0.40) can never "win" even when it's the real
    # driver, since OSF/HDF/PWF can reach 1.0. Report co-drivers when the
    # top two are close: forcing a single label when the model genuinely
    # can't distinguish (e.g. TWF's hidden per-unit randomness vs. a
    # moderately-elevated OSF ratio on the same row) is a false certainty.
    max_possible = {"OSF": 1.0, "HDF": 1.0, "PWF": 1.0, "TWF": p["twf_max_risk"]}
    normalized = {k: risks[k] / max_possible[k] for k in max_possible}
    ranked = sorted(normalized.items(), key=lambda kv: -kv[1])
    top_mode, top_val = ranked[0]
    second_mode, second_val = ranked[1]

    CO_DRIVER_MARGIN = 0.15
    if top_val <= 0.01:
        dominant_label = "none"
    elif (top_val - second_val) <= CO_DRIVER_MARGIN and second_val > 0.01:
        dominant_label = f"{top_mode}/{second_mode}"
    else:
        dominant_label = top_mode

    return {
        "health_score": round(health, 1),
        "p_fail_composite": round(p_fail, 4),
        "risk_OSF": round(r_osf, 3),
        "risk_HDF": round(r_hdf, 3),
        "risk_PWF": round(r_pwf, 3),
        "risk_TWF": round(r_twf, 3),
        "risk_RNF_baseline": r_rnf,
        "osf_ratio_to_threshold": round(osf_ratio, 3),
        "dominant_mode": dominant_label,
        "dominant_mode_normalized_score": round(top_val, 3),
    }


if __name__ == "__main__":
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

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # Hand-pick a spread of illustrative rows rather than random sampling,
    # so we can sanity-check the model against known ground truth.
    picks = []

    # 1. A comfortably healthy row (low wear, mid-range everything, no flags).
    healthy = df[(df["failure"] == 0) & (df["wear"] < 30)].iloc[0]
    picks.append(("healthy / low wear", healthy))

    # 2-4. One actual OSF failure per Type, to check the model flags all three
    # correctly despite the different thresholds.
    for t in ["L", "M", "H"]:
        subset = df[(df["OSF"] == 1) & (df["Type"] == t)]
        if len(subset) > 0:
            picks.append((f"actual OSF failure, Type {t}", subset.iloc[0]))

    # 5. A row just BELOW the OSF threshold -- the early-warning case this
    # model exists to catch (not flagged as failure in the raw data, but
    # should show elevated risk_OSF).
    df["twp_tmp"] = df["torque"] * df["wear"]
    df["osf_thr_tmp"] = df["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    near_osf = df[
        (df["OSF"] == 0)
        & (df["twp_tmp"] / df["osf_thr_tmp"] > 0.85)
        & (df["twp_tmp"] / df["osf_thr_tmp"] < 1.0)
    ]
    if len(near_osf) > 0:
        picks.append(("approaching OSF threshold, not yet failed", near_osf.iloc[0]))

    # 6. An actual HDF failure.
    hdf_fail = df[df["HDF"] == 1].iloc[0]
    picks.append(("actual HDF failure", hdf_fail))

    # 7. An actual PWF failure.
    pwf_fail = df[df["PWF"] == 1].iloc[0]
    picks.append(("actual PWF failure", pwf_fail))

    # 8. An actual TWF failure.
    twf_fail = df[df["TWF"] == 1].iloc[0]
    picks.append(("actual TWF failure", twf_fail))

    # 9. A TWF-window row that did NOT fail (wear in [200,240], TWF=0) --
    # tests that capped/moderate risk behavior, not false certainty.
    twf_survivor = df[(df["wear"].between(200, 240)) & (df["TWF"] == 0)].iloc[0]
    picks.append(("in TWF wear window, did NOT fail", twf_survivor))

    # 10. One of the RNF-only near-miss rows -- should show near-zero health
    # penalty from the model (baseline only), demonstrating we correctly do
    # NOT treat RNF as a predictive signal.
    near_miss_raw = pd.read_csv("data/near_miss_events.csv").iloc[0]
    near_miss = near_miss_raw.rename({
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "wear",
        "Machine failure": "failure",
    })
    picks.append(("RNF near-miss (no real precursor)", near_miss))

    rows_out = []
    for label, row in picks:
        row_std = {
            "air_temp": row["air_temp"],
            "process_temp": row["process_temp"],
            "rpm": row["rpm"],
            "torque": row["torque"],
            "wear": row["wear"],
            "Type": row["Type"],
        }
        result = compute_health(row_std)
        rows_out.append({
            "case": label,
            "UDI": row["UDI"],
            "Type": row["Type"],
            "torque": row["torque"],
            "wear": row["wear"],
            "actual_failure": row["failure"],
            "actual_modes": "+".join(m for m in ["TWF","HDF","PWF","OSF","RNF"] if row[m] == 1) or "-",
            **result,
        })

    out_df = pd.DataFrame(rows_out)
    print(out_df.to_string(index=False))
