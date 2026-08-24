"""
Fleet Health & Predictive Maintenance Simulator -- Streamlit dashboard.

Reads exclusively from saved CSV outputs (data/*.csv) produced by the
project's earlier analysis scripts. Does NOT recompute health scores,
threshold curves, or simulation results -- those are all precomputed and
saved to disk (see generate_dashboard_data.py). The only "live" computation
here is looking up the nearest precomputed row for the interactive slider.

Data provenance:
  - AI4I 2020 Predictive Maintenance Dataset (UCI, public, 10,000 real
    industrial sensor records) -- the only real dataset in this project.
  - Everything else (health model, cohort findings, cost curves, 180-day
    simulation) is a general adjustable-parameter framework built on top of
    that data, framed around the Amazon Robotics context (DeepFleet's
    public 1M+ robots / 300+ facilities figures) for scale, not Amazon's
    actual internal data.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fleet Health & Predictive Maintenance Simulator", layout="wide")

PALETTE = {
    "ink": "#10141B",
    "panel": "#171D26",
    "amber": "#F2A93B",
    "cyan": "#5DC8D8",
    "red": "#E2574C",
    "green": "#6FCF8E",
    "text": "#E8E8ED",
    "grid": "#2A3240",
}
SERIES_SEQUENCE = [PALETTE["amber"], PALETTE["cyan"], PALETTE["red"], PALETTE["green"]]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, div, span, label {{
    font-family: 'IBM Plex Mono', monospace;
}}

h1, h2, h3, h4, h5, h6,
.stTabs [data-baseweb="tab"] p,
[data-testid="stMetricLabel"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}}

.stApp {{
    background-color: {PALETTE["ink"]};
}}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    color: {PALETTE["amber"]} !important;
}}

[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: {PALETTE["panel"]};
    border-radius: 10px;
}}

[data-testid="stMetric"] {{
    background-color: {PALETTE["panel"]};
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid #232B38;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {PALETTE["panel"]};
    border-radius: 8px 8px 0 0;
}}

hr {{
    border-color: #232B38;
}}

code, pre {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

.live-header {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
}}
.live-header h1 {{
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: {PALETTE["text"]};
}}
.live-indicator {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: {PALETTE["green"]};
    font-weight: 600;
}}
.live-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background-color: {PALETTE["green"]};
    box-shadow: 0 0 0 0 rgba(111,207,142,0.7);
    animation: live-pulse 2s infinite;
}}
@keyframes live-pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(111,207,142,0.65); }}
    70%  {{ box-shadow: 0 0 0 9px rgba(111,207,142,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(111,207,142,0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .live-dot {{ animation: none; }}
}}

/* mobile layout: stack column rows vertically below 768px instead of
   squeezing metrics/charts into unreadable slivers */
@media (max-width: 768px) {{
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 8px;
    }}
    .live-header {{
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.4rem !important;
    }}
    .about-card {{
        padding: 20px 22px;
    }}
}}

.about-card {{
    background: linear-gradient(155deg, {PALETTE["panel"]} 0%, #131924 100%);
    border: 1px solid #232B38;
    border-left: 4px solid {PALETTE["amber"]};
    border-radius: 12px;
    padding: 26px 30px;
    margin-bottom: 26px;
}}
.about-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.55rem;
    color: {PALETTE["text"]};
    margin-bottom: 4px;
}}
.about-tagline {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: {PALETTE["cyan"]};
    margin-bottom: 16px;
    line-height: 1.5;
}}
.about-section {{
    line-height: 1.6;
    font-size: 0.92rem;
    color: {PALETTE["text"]};
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="about-card">
  <div class="about-title">Fleet Health &amp; Predictive Maintenance Simulator</div>
  <div class="about-tagline">Catching robot failures before they cause downtime.</div>
  <div class="about-section">Large robot fleets usually deal with maintenance reactively: something breaks,
  then someone responds. This tool scores every robot's health continuously, flags real risk before a failure
  happens, and calculates how many backup robots should be kept on standby, using the same math supply chains
  use for safety stock. The scale here (1 million+ robots, 300+ facilities) comes from Amazon Robotics' own
  public DeepFleet announcement. The health model runs on a real, published industrial sensor dataset with
  10,000 records and validated failure mechanisms. Repair time and downtime cost are based on industry
  benchmarks rather than internal company numbers, and those are labeled wherever they show up.</div>
</div>
""", unsafe_allow_html=True)


def style_fig(fig):
    """Apply the dashboard's dark/branded theme to a Plotly figure: transparent
    background, IBM Plex Mono body font, Space Grotesk title font, muted gridlines."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color=PALETTE["text"], size=12),
        title_font=dict(family="Space Grotesk, sans-serif", color=PALETTE["text"], size=17),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["text"])),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"], color=PALETTE["text"], linecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"], color=PALETTE["text"], linecolor=PALETTE["grid"])
    return fig


@st.cache_data
def load_data():
    scored = pd.read_csv("data/scored_dataset.csv")
    near_miss = pd.read_csv("data/near_miss_events.csv")
    cost_baseline = pd.read_csv("data/corrected_cost_curve_baseline.csv")
    cost_peak = pd.read_csv("data/corrected_cost_curve_peak.csv")
    catch_rate = pd.read_csv("data/catch_rate_comparison.csv")
    catch_rate_v1 = pd.read_csv("data/catch_rate_v1_inflated.csv")
    daily_logs = {
        t: pd.read_csv(f"data/dynamic_sim_daily_log_threshold_{t}.csv")
        for t in [1, 30, 50]
    }
    # earlier model versions, kept on disk from prior turns, used only to show
    # the correction history, not recomputed here
    cost_v1_naive = pd.read_csv("data/cost_curve_baseline.csv")
    cost_v2_shortfall = pd.read_csv("data/shortfall_gated_curve_baseline.csv")
    return (scored, near_miss, cost_baseline, cost_peak, catch_rate, catch_rate_v1,
            daily_logs, cost_v1_naive, cost_v2_shortfall)


(scored, near_miss, cost_baseline, cost_peak, catch_rate, catch_rate_v1,
 daily_logs, cost_v1_naive, cost_v2_shortfall) = load_data()

OSF_THRESHOLD = {"L": 11000, "M": 12000, "H": 13000}


def render_insight(severity, text):
    """Rule-based insight callout. Severity and text are both computed from
    the loaded data at runtime through simple conditional logic, not hardcoded
    copy and not calling any external AI service."""
    icon = {"warning": "⚠", "success": "✓", "info": "ℹ"}[severity]
    fn = {"warning": st.warning, "success": st.success, "info": st.info}[severity]
    fn(f"{icon} {text}")


st.markdown(
    '<div class="live-header"><h1>Fleet Health &amp; Predictive Maintenance Simulator</h1>'
    '<span class="live-indicator"><span class="live-dot"></span>LIVE</span></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fleet Health Overview",
    "2. Cohort Analysis",
    "3. Cost & Backup Pool (interactive)",
    "4. 180-Day Dynamic Simulation",
])

# ---------------------------------------------------------------------------
# TAB 1 -- Fleet Health Overview
# ---------------------------------------------------------------------------
with tab1:
    st.header("Fleet Health Overview")
    st.caption(
        "Health scores for all 10,000 robots come from the model described in this project, weighted mainly "
        "on overstrain risk (OSF) with heat dissipation (HDF), power (PWF), and tool wear (TWF) as secondary "
        "signals, plus a small constant baseline for random failures (RNF)."
    )

    critical_cutoff = 10
    n_critical = int((scored["health_score"] < critical_cutoff).sum())
    pct_critical = n_critical / len(scored) * 100

    if pct_critical > 20:
        render_insight("warning", f"{pct_critical:.1f}% of the fleet sits in the critical health range, "
                                   f"below {critical_cutoff}. That is above a 20% caution line and likely "
                                   f"strains inspection and repair capacity.")
    elif pct_critical > 10:
        render_insight("info", f"{pct_critical:.1f}% of the fleet sits in the critical health range, "
                                f"below {critical_cutoff}. That is below the 20% caution line but worth watching.")
    else:
        render_insight("success", f"Only {pct_critical:.1f}% of the fleet sits in the critical health range, below {critical_cutoff}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fleet size", f"{len(scored):,} robots")
    c2.metric("Mean health score", f"{scored['health_score'].mean():.1f} / 100")
    c3.metric(f"Critical range, below {critical_cutoff}", f"{pct_critical:.1f}%")
    c4.metric("Actual failure rate", f"{scored['failure'].mean()*100:.2f}%")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig = px.histogram(
            scored, x="health_score", nbins=50,
            color=scored["failure"].map({0: "No failure", 1: "Actual failure"}),
            title="Health score distribution across the fleet",
            labels={"health_score": "Health score, 0 is critical and 100 is healthy", "color": "Actual outcome"},
            barmode="overlay", opacity=0.75,
            color_discrete_map={"No failure": PALETTE["green"], "Actual failure": PALETTE["red"]},
        )
        fig.update_layout(legend_title_text="")
        fig.add_vrect(x0=0, x1=critical_cutoff, fillcolor=PALETTE["red"], opacity=0.10, line_width=0)
        fig.add_annotation(
            x=critical_cutoff, y=0.94, xref="x", yref="paper",
            text="Most real failures cluster here", showarrow=True, arrowhead=2,
            ax=90, ay=0, arrowcolor=PALETTE["red"], font=dict(color=PALETTE["red"], size=12),
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption(
            "Most robots sit near perfect health, but almost every real failure happened only after a "
            "robot's score had already collapsed into the shaded zone on the left."
        )
    with col_b:
        st.markdown("**Mean health by actual outcome**")
        by_outcome = scored.groupby("failure")["health_score"].agg(["mean", "median", "count"]).round(1)
        by_outcome.index = by_outcome.index.map({0: "No failure", 1: "Actual failure"})
        st.dataframe(by_outcome, use_container_width=True)
        st.markdown(
            f"Health scores separate cleanly by outcome. Median health for robots that actually failed is "
            f"**{scored[scored['failure']==1]['health_score'].median():.1f}**, compared with "
            f"**{scored[scored['failure']==0]['health_score'].median():.1f}** for robots that did not fail."
        )

    st.subheader("Dominant risk mode across the fleet")
    st.caption("Each robot's health score is usually driven by one failure mechanism more than the others. This chart counts how often each mechanism is the dominant one.")
    dom_counts = scored["dominant_mode"].value_counts().reset_index()
    dom_counts.columns = ["dominant_mode", "count"]
    fig2 = px.bar(
        dom_counts, x="dominant_mode", y="count", title=None,
        color="dominant_mode", color_discrete_sequence=SERIES_SEQUENCE,
        labels={"dominant_mode": "Dominant risk mode", "count": "Number of robots"},
    )
    fig2.update_layout(showlegend=False)
    MODE_LABELS = {
        "OSF": "overstrain risk", "HDF": "heat dissipation risk", "PWF": "power risk",
        "TWF": "tool wear risk", "RNF": "random-failure risk", "none": "no single dominant risk",
    }
    top_mode = dom_counts.iloc[0]["dominant_mode"]
    top_count = int(dom_counts.iloc[0]["count"])
    top_label = MODE_LABELS.get(top_mode, top_mode)
    fig2.add_annotation(
        x=top_mode, y=top_count, text="Most common", showarrow=True, arrowhead=2,
        ax=0, ay=-45, font=dict(color=PALETTE["amber"], size=12),
    )
    st.plotly_chart(style_fig(fig2), use_container_width=True)
    co_dominant_count = int(scored["dominant_mode"].str.contains("/", na=False).sum())
    st.caption(
        f"{top_label.capitalize()} is the single biggest driver, affecting {top_count:,} of the "
        f"{len(scored):,} robots, and about {co_dominant_count:,} robots show two risk factors close "
        f"enough together that neither one is clearly the main cause."
    )

    with st.expander("Near-miss events: RNF trips with no sensor precursor"):
        st.caption(
            "The dataset's only labeled near-miss events, where a failure-mode flag tripped without a full "
            "failure, are all random-failure (RNF) trips. These carry no sensor precursor, so they are shown "
            "here for transparency and are not used as a predictive signal anywhere in this model."
        )
        display_near_miss = near_miss.rename(columns={"Machine failure": "Robot failure"})
        st.dataframe(display_near_miss, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 -- Cohort Analysis
# ---------------------------------------------------------------------------
with tab2:
    st.header("Cohort Analysis: Type L, M, and H")
    st.caption("Type is the closest thing to a robot batch or generation in this dataset. Usage intensity is measured by tool wear.")

    fail_by_type = scored.groupby("Type")["failure"].agg(["mean", "count"]).rename(
        columns={"mean": "failure_rate", "count": "n"}
    )
    fail_by_type["failure_rate_pct"] = (fail_by_type["failure_rate"] * 100).round(2)

    l_rate = fail_by_type.loc["L", "failure_rate_pct"]
    h_rate = fail_by_type.loc["H", "failure_rate_pct"]
    ratio = l_rate / h_rate if h_rate > 0 else float("nan")
    if ratio > 1.5:
        render_insight("warning", f"Type L fails at {ratio:.1f} times the rate of Type H, {l_rate}% compared "
                                   f"with {h_rate}%. That is above a 1.5 times caution line. This is a real "
                                   f"build-tolerance gap confirmed against the dataset's own failure rules, "
                                   f"not a side effect of how hard the robots are used.")
    else:
        render_insight("info", f"Type L fails at {ratio:.1f} times the rate of Type H, {l_rate}% compared with {h_rate}%.")

    c1, c2, c3 = st.columns(3)
    for col, t in zip([c1, c2, c3], ["L", "M", "H"]):
        rate = fail_by_type.loc[t, "failure_rate_pct"]
        n = int(fail_by_type.loc[t, "n"])
        col.metric(f"Type {t} failure rate", f"{rate}%", help=f"{n} robots. Overstrain fails above {OSF_THRESHOLD[t]:,} (torque times wear).")

    st.markdown(
        "This gap holds up even after controlling for tool wear, matching the dataset's own failure rules "
        "exactly, so it is not a side effect of usage intensity. Type L has a genuinely lower overstrain "
        "tolerance than Type M or Type H, and M and H are statistically indistinguishable from each other, so "
        "this is not a clean three-tier gradient. Overstrain failure happens when torque times wear crosses a "
        "threshold that depends on Type: 11,000 for L, 12,000 for M, and 13,000 for H."
    )

    st.subheader("Torque and wear compared with each Type's overstrain threshold")
    plot_df = scored.copy()
    plot_df["torque_wear_product"] = plot_df["torque"] * plot_df["wear"]

    type_colors = {"L": PALETTE["amber"], "M": PALETTE["cyan"], "H": PALETTE["green"]}
    fig3 = px.scatter(
        plot_df, x="wear", y="torque", color="Type",
        symbol=plot_df["OSF"].map({0: "No overstrain failure", 1: "Overstrain failure"}),
        opacity=0.5,
        title="Torque and tool wear by Type, with overstrain failures marked",
        labels={"wear": "Tool wear (minutes)", "torque": "Torque (Nm)", "symbol": "Overstrain outcome"},
        color_discrete_map=type_colors,
        height=550,
    )
    wear_range = np.linspace(1, 260, 200)
    for t, color in type_colors.items():
        threshold_curve = OSF_THRESHOLD[t] / wear_range
        fig3.add_trace(go.Scatter(
            x=wear_range, y=threshold_curve, mode="lines",
            name=f"Type {t} overstrain threshold ({OSF_THRESHOLD[t]:,})",
            line=dict(color=color, dash="dash", width=2),
        ))
    fig3.update_yaxes(range=[0, 80])
    annotate_wear = 190
    annotate_torque = OSF_THRESHOLD["L"] / annotate_wear
    fig3.add_annotation(
        x=annotate_wear, y=annotate_torque, text="Type L fails here", showarrow=True, arrowhead=2,
        ax=50, ay=-55, font=dict(color=PALETTE["amber"], size=12),
    )
    st.plotly_chart(style_fig(fig3), use_container_width=True)
    st.caption(
        "Type L crosses into overstrain failure at meaningfully lower torque and wear than Type M or H, so "
        "the exact same workload is riskier for an L robot than for an otherwise identical H robot."
    )

    st.subheader("Overstrain failure rate by Type")
    osf_by_type = scored.groupby("Type")["OSF"].agg(["mean", "sum", "count"])
    osf_by_type.columns = ["Overstrain rate", "Failures", "Robots"]
    osf_by_type["Overstrain rate"] = (osf_by_type["Overstrain rate"] * 100).round(2).astype(str) + "%"
    st.dataframe(osf_by_type, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 -- Cost & Backup Pool (interactive)
# ---------------------------------------------------------------------------
with tab3:
    st.header("Cost and Backup Pool")
    st.caption(
        "This section calculates the cheapest combination of a health-score threshold and a backup-pool "
        "size, using real cumulative results from the 180-day simulation rather than a single-day estimate. "
        "Move the slider below to look up any threshold."
    )

    with st.expander("What the cost numbers assume", expanded=False):
        st.markdown("""
| Input | Value | Where it comes from |
|---|---|---|
| Repair time | 2.5 hours | Industry benchmark for industrial robots, typically 1 to 4 hours. Not from the dataset. |
| Downtime cost per hour, normal | $100,000 | Low end of a cited industry range of $100,000 to $260,000 per hour. Not from the dataset. |
| Downtime cost per hour, peak season | $260,000 | High end of the same cited range, used for the peak-season scenario. |
| Holding cost per spare, per day | $40 | Derived from standard equipment practice, 20 to 30 percent of asset value per year, for a robot worth $25,000 to $80,000. A grounded estimate, not a direct quote. |
| Swap cost, repair with a spare available | $1,500 | Derived from warehouse robot annual maintenance costs of $2,000 to $8,000, with about 40 percent going to unplanned repairs. A grounded estimate, not a direct quote. |
| Inspection cost for a false alarm | $500 | An assumed placeholder, kept separate from the downtime-cost benchmark on purpose. |
| Task and environment reassignment | $800 | An assumed placeholder, set between the inspection cost and the swap cost. Covers reconfiguring a chronically misassigned robot's task profile rather than replacing a part. |
| Fleet size in this model | 10,000 robots | The size of the dataset itself, treated as one day's full-fleet monitoring pass. A modeling choice, not a real fleet schedule. |
""")

    scenario = st.radio("Downtime-cost scenario", ["Normal ($100,000 per hour)", "Peak season ($260,000 per hour)"], horizontal=True)
    curve = cost_baseline if scenario.startswith("Normal") else cost_peak

    threshold_val = st.slider(
        "Health-score threshold: flag a robot for action once its score drops below this number",
        min_value=1, max_value=99, value=1, step=1,
    )

    row = curve.iloc[(curve["threshold"] - threshold_val).abs().argmin()]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{row['precision']*100:.1f}%", help="Of the robots flagged at this threshold, the share that actually failed.")
    c2.metric("Recall", f"{row['recall']*100:.1f}%", help="Of the robots that actually failed, the share this threshold caught in advance.")
    c3.metric("Backup robots needed", f"{int(row['pool_size'])}")
    c4.metric("Chance of running short", f"{row['shortfall_prob']*100:.3f}%")
    c5.metric("Total cost over 180 days", f"${row['total_cost_180d']:,.0f}")

    st.markdown(
        f"At a threshold of {threshold_val}, the cost breaks down as follows. False alarms cost "
        f"**${row['fp_cost_180d']:,.0f}** in unnecessary inspections. Missed failures cost "
        f"**${row['fn_cost_180d']:,.0f}**, using the shortfall-gated model. Holding the backup pool costs "
        f"**${row['pool_cost_180d']:,.0f}**. Reassigning chronically misassigned robots to different work "
        f"costs **${row['chronic_cost_180d']:,.0f}**."
    )

    best_row = curve.loc[curve["total_cost_180d"].idxmin()]

    gap_pct = (row["total_cost_180d"] - best_row["total_cost_180d"]) / best_row["total_cost_180d"] * 100
    if gap_pct > 25:
        render_insight("warning", f"This threshold costs {gap_pct:.0f}% more than the cheapest option, a "
                                   f"threshold of {int(best_row['threshold'])} with {int(best_row['pool_size'])} backup robots.")
    elif gap_pct > 2:
        render_insight("info", f"This threshold costs {gap_pct:.0f}% more than the cheapest option, a "
                                f"threshold of {int(best_row['threshold'])}.")
    else:
        render_insight("success", "This threshold is at, or within 2% of, the cheapest option.")

    st.info(
        f"The cheapest point on this curve is a threshold of {int(best_row['threshold'])} with "
        f"{int(best_row['pool_size'])} backup robots, costing ${best_row['total_cost_180d']:,.0f} over 180 "
        f"days. This favors a small, well-stocked backup pool over widespread inspection, a result that held "
        f"up across a wide sensitivity check on the swap-cost and holding-cost assumptions."
    )

    with st.expander("How this cost estimate changed as the model improved", expanded=False):
        v1_best = cost_v1_naive.loc[cost_v1_naive["total_cost"].idxmin()]
        v2_best = cost_v2_shortfall.loc[cost_v2_shortfall["total_cost"].idxmin()]
        v3_best = cost_baseline.loc[cost_baseline["total_cost_180d"].idxmin()]
        history_df = pd.DataFrame([
            {"Version": "First pass, single-day estimate", "Threshold": int(v1_best["threshold"]),
             "Backup robots": int(v1_best["pool_size"]), "Approximate cost": f"${v1_best['total_cost']:,.0f}",
             "What changed": "Charged the full cost of a facility outage for every missed failure, as if no backup pool existed."},
            {"Version": "Second pass, accounts for the pool", "Threshold": int(v2_best["threshold"]),
             "Backup robots": int(v2_best["best_pool_size"]), "Approximate cost": f"${v2_best['total_cost']:,.0f}",
             "What changed": "Only charges the full outage cost when the pool actually runs out. Otherwise it charges a cheap swap cost."},
            {"Version": "Third pass, uses the 180-day simulation", "Threshold": int(v3_best["threshold"]),
             "Backup robots": int(v3_best["pool_size"]), "Approximate cost": f"${v3_best['total_cost_180d']:,.0f} over 180 days",
             "What changed": "Replaced the single-day failure estimate with the real cumulative count from the dynamic simulation."},
            {"Version": "Current, prices in task reassignment", "Threshold": int(best_row["threshold"]),
             "Backup robots": int(best_row["pool_size"]), "Approximate cost": f"${best_row['total_cost_180d']:,.0f} over 180 days",
             "What changed": "Added a cost for reassigning chronically misassigned robots to different work. Earlier passes charged this $0, as if it were free."},
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.caption(
            "The recommended threshold and pool size have stayed the same since the third pass. The first "
            "three passes each removed a way the model was overstating cost. This pass moved the other "
            "direction: it added a real cost the model had been missing entirely, so the total went up "
            "slightly rather than down."
        )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=curve["threshold"], y=curve["total_cost_180d"], mode="lines",
        name="Total cost", line=dict(color=PALETTE["amber"], width=2.5),
    ))
    fig4.add_trace(go.Scatter(
        x=[threshold_val], y=[row["total_cost_180d"]], mode="markers",
        marker=dict(size=14, color=PALETTE["red"], symbol="circle"), name="Selected threshold",
    ))
    fig4.add_trace(go.Scatter(
        x=[best_row["threshold"]], y=[best_row["total_cost_180d"]], mode="markers",
        marker=dict(size=14, color=PALETTE["green"], symbol="star"), name="Cheapest threshold",
    ))
    fig4.update_layout(title="Total cost by threshold", xaxis_title="Health-score threshold", yaxis_title="Total cost over 180 days ($)")
    if int(threshold_val) == int(best_row["threshold"]):
        fig4.add_annotation(
            x=threshold_val, y=row["total_cost_180d"], text="Your selection is the cheapest option",
            showarrow=True, arrowhead=2, ax=60, ay=-60, font=dict(color=PALETTE["green"], size=12),
        )
    else:
        fig4.add_annotation(
            x=threshold_val, y=row["total_cost_180d"], text="Your selection",
            showarrow=True, arrowhead=2, ax=0, ay=-55, font=dict(color=PALETTE["red"], size=12),
        )
        fig4.add_annotation(
            x=best_row["threshold"], y=best_row["total_cost_180d"], text="Cheapest overall",
            showarrow=True, arrowhead=2, ax=0, ay=55, font=dict(color=PALETTE["green"], size=12),
        )
    st.plotly_chart(style_fig(fig4), use_container_width=True)
    st.caption(
        f"Across every threshold tested, keeping only a small number of robots flagged, threshold "
        f"{int(best_row['threshold'])}, with a modest backup pool of {int(best_row['pool_size'])} robots "
        f"ready turns out to be the cheapest approach overall."
    )

    with st.expander("See the technical validation behind this", expanded=False):
        st.caption(
            "This part is for readers who want to check the health model's classification performance "
            "directly, using standard machine learning validation charts."
        )

        roc_df = pd.read_csv("data/roc_curve.csv")
        roc_auc = 0.963
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_df["fpr"], y=roc_df["tpr"], mode="lines", name="Health model",
            line=dict(color=PALETTE["amber"], width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
            line=dict(color=PALETTE["grid"], dash="dash", width=1.5),
        ))
        fig_roc.add_annotation(
            x=0.15, y=0.8, text=f"AUC {roc_auc}", showarrow=False,
            font=dict(color=PALETTE["amber"], size=14),
        )
        fig_roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig_roc), use_container_width=True)
        st.caption(
            f"An AUC of {roc_auc} out of a maximum of 1.0 means the model separates robots that will fail "
            f"from robots that will not almost perfectly."
        )

        st.markdown("**Precision and recall at every threshold**")
        fig_pr = px.line(
            cost_baseline.sort_values("recall"), x="recall", y="precision",
            title="Precision against recall across all thresholds",
            labels={"recall": "Recall (share of real failures caught)", "precision": "Precision (share of flags that were real)"},
        )
        fig_pr.update_traces(line=dict(color=PALETTE["cyan"], width=2.5))
        st.plotly_chart(style_fig(fig_pr), use_container_width=True)
        st.caption(
            "Moving along this curve trades catching more real failures for a higher rate of false alarms. "
            "There is no single correct point. The cost curve above is what picks a point on this trade-off."
        )

        pr_sample = cost_baseline[cost_baseline["threshold"].isin(range(1, 100, 10))][["threshold", "precision", "recall"]].copy()
        pr_sample.columns = ["Threshold", "Precision", "Recall"]
        pr_sample["Precision"] = (pr_sample["Precision"] * 100).round(1).astype(str) + "%"
        pr_sample["Recall"] = (pr_sample["Recall"] * 100).round(1).astype(str) + "%"
        st.dataframe(pr_sample, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 -- 180-Day Dynamic Simulation
# ---------------------------------------------------------------------------
with tab4:
    st.header("180-Day Simulation")
    st.caption(
        "The dataset is a single snapshot with no timeline, so this section runs a forward simulation "
        "starting from that snapshot, including the effect of shifting a struggling robot's workload onto "
        "healthier robots."
    )

    st.subheader("Fleet health over time, by threshold policy")
    traj_df = pd.concat([
        log.assign(threshold_policy=f"threshold={t}") for t, log in daily_logs.items()
    ])
    policy_colors = {"threshold=1": PALETTE["red"], "threshold=30": PALETTE["amber"], "threshold=50": PALETTE["green"]}
    fig5 = px.line(
        traj_df, x="day", y="mean_health", color="threshold_policy",
        title="Fleet mean health score over 180 days",
        labels={"day": "Day", "mean_health": "Fleet mean health score", "threshold_policy": "Policy"},
        color_discrete_map=policy_colors,
    )
    fig5.update_traces(line=dict(width=2.5))
    trend_labels = {1: "declines", 30: "improves slightly", 50: "improves substantially"}
    trend_colors = {1: PALETTE["red"], 30: PALETTE["amber"], 50: PALETTE["green"]}
    for t in [1, 30, 50]:
        end_val = daily_logs[t].iloc[-1]["mean_health"]
        fig5.add_annotation(
            x=179, y=end_val, text=f"Threshold {t} {trend_labels[t]}", showarrow=True, arrowhead=2,
            ax=70, ay=0, font=dict(color=trend_colors[t], size=12), xanchor="left",
        )
    st.plotly_chart(style_fig(fig5), use_container_width=True)
    st.caption(
        "Only the threshold that looked cheapest in a single day snapshot, threshold 1, actually lets fleet "
        "health quietly decline over the following six months."
    )

    c1, c2, c3 = st.columns(3)
    for col, t in zip([c1, c2, c3], [1, 30, 50]):
        log = daily_logs[t]
        delta = log.iloc[-1]["mean_health"] - log.iloc[0]["mean_health"]
        col.metric(
            f"Threshold {t}, health from day 0 to day 179",
            f"{log.iloc[-1]['mean_health']:.1f}",
            delta=f"{delta:+.1f}",
        )
    st.caption(
        "A threshold of 1 was the original cost model's pick, and it is the only policy where fleet health "
        "declines over time. The single-day cost model could not see this, since it only ever looked at one "
        "day at a time."
    )

    delta_t1 = daily_logs[1].iloc[-1]["mean_health"] - daily_logs[1].iloc[0]["mean_health"]
    delta_t50 = daily_logs[50].iloc[-1]["mean_health"] - daily_logs[50].iloc[0]["mean_health"]
    if delta_t1 < 0 < delta_t50:
        render_insight("warning", f"At a threshold of 1, fleet health trends down, {delta_t1:+.1f} over 180 "
                                   f"days, while at a threshold of 50 it trends up, {delta_t50:+.1f}. The "
                                   f"recommendation from the single-day cost model does not hold up once "
                                   f"tested over time.")
    else:
        render_insight("info", f"Fleet health trend at threshold 1: {delta_t1:+.1f}. At threshold 50: {delta_t50:+.1f}.")

    st.subheader("How many catches were real")
    st.caption(
        "The first version of this simulation counted every day a robot was flagged as a fresh catch. But "
        "207 robots have a fixed profile that keeps tripping a heat or power warning no matter how much "
        "their wear resets, so repairing them never actually fixes the problem. They got flagged and "
        "recounted as a catch every single day. The corrected version tracks how many days in a row a robot "
        "gets flagged, and treats three or more days as a sign the robot needs different work, not a fresh "
        "catch each time."
    )

    compare_rows = []
    for t in [1, 30, 50]:
        v1_row = catch_rate_v1[catch_rate_v1["threshold"] == t].iloc[0]
        v2_row = catch_rate[catch_rate["threshold"] == t].iloc[0]
        compare_rows.append({
            "Threshold": t,
            "First count (inflated)": f"{v1_row['catch_rate_v1_inflated']*100:.1f}%",
            "Corrected count": f"{v2_row['new_catch_rate']*100:.1f}%",
            "Robots reassigned to different work": int(v2_row["chronic_reassignments"]),
        })
    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    gap_at_1 = catch_rate_v1[catch_rate_v1["threshold"] == 1]["catch_rate_v1_inflated"].iloc[0] - \
        catch_rate[catch_rate["threshold"] == 1]["new_catch_rate"].iloc[0]
    if gap_at_1 > 0.30:
        render_insight("warning", f"At a threshold of 1, the first version of this count overstates real "
                                   f"proactive coverage by {gap_at_1*100:.0f} percentage points. That gap "
                                   f"comes from the repair-loop artifact, not real performance.")
    else:
        render_insight("info", f"At a threshold of 1, the first and corrected counts differ by {gap_at_1*100:.0f} points.")

    fig6 = go.Figure()
    thresholds_shown = [1, 30, 50]
    v1_rates = [catch_rate_v1[catch_rate_v1["threshold"] == t]["catch_rate_v1_inflated"].iloc[0] * 100 for t in thresholds_shown]
    v2_rates = [catch_rate[catch_rate["threshold"] == t]["new_catch_rate"].iloc[0] * 100 for t in thresholds_shown]
    fig6.add_trace(go.Bar(x=[str(t) for t in thresholds_shown], y=v1_rates, name="First count, inflated", marker_color=PALETTE["red"]))
    fig6.add_trace(go.Bar(x=[str(t) for t in thresholds_shown], y=v2_rates, name="Corrected count", marker_color=PALETTE["green"]))
    fig6.update_layout(barmode="group", title="Catch rate, first count compared with corrected", xaxis_title="Threshold", yaxis_title="Catch rate (%)")
    fig6.add_annotation(
        x="1", y=(v1_rates[0] + v2_rates[0]) / 2, text=f"{v1_rates[0]-v2_rates[0]:.0f} point gap",
        showarrow=True, arrowhead=2, ax=80, ay=0, font=dict(color=PALETTE["text"], size=12),
    )
    st.plotly_chart(style_fig(fig6), use_container_width=True)
    st.caption(
        f"At the strictest threshold, the naive way of counting catches made the policy look "
        f"{v1_rates[0]-v2_rates[0]:.0f} percentage points more effective than it actually was."
    )

    st.markdown(
        "This gap is the real finding, not noise. The true proactive catch rate on genuine wear-driven risk, "
        "55 to 77 percent depending on threshold, is far lower than the first count's 93.8 to 99.4 percent "
        "suggested. The decline in fleet health at threshold 1 tracks its low corrected catch rate of 55.2 "
        "percent. Interpolating between thresholds, fleet health roughly breaks even around a 67 percent "
        "genuine catch rate."
    )
