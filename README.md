# Fleet Health & Backup-Pool Simulator

An adjustable-parameter framework for robot fleet health scoring, cohort reliability tracking, and
backup-pool/spares provisioning economics, framed around the Amazon Robotics context.

**Data provenance -- read this first:**
- The only real dataset used is the **AI4I 2020 Predictive Maintenance Dataset** (UCI, public,
  10,000 real industrial sensor records). All health-score modeling, cohort findings, and threshold
  validation are derived from and checked against this dataset's actual (published) failure-generating
  rules.
- Amazon's fleet **scale** (1M+ robots, 300+ facilities) is DeepFleet's public 2025 figure, used only
  for framing context.
- **This is not Amazon's internal data or an Amazon-endorsed tool.** Repair time, downtime cost,
  holding cost, and swap cost are either cited industry benchmarks or explicitly-labeled derived
  assumptions -- all called out by source in the dashboard itself (Tab 3 expander).

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Theme (dark, amber/cyan/red/green) is set in `.streamlit/config.toml`;
fonts (Space Grotesk / IBM Plex Mono) are injected via CSS in `app.py` since Streamlit's native theming
doesn't support custom font families.

## Dashboard sections

1. **Fleet Health Overview** -- health-score distribution, critical-range %, dominant failure mode
2. **Cohort Analysis** -- Type L/M/H comparison, with the validated finding that L's lower overstrain
   (OSF) tolerance survives controlling for usage intensity (it's a real build-tolerance gap, not a
   wear-intensity artifact)
3. **Cost & Backup Pool (interactive)** -- live threshold slider over a precomputed, shortfall-gated,
   jointly-optimized (threshold, pool_size) cost curve, plus a correction-history panel showing how the
   recommended pool size changed across three modeling iterations
4. **180-Day Dynamic Simulation** -- forward-simulated fleet health trajectories under different
   reassignment thresholds, and the corrected-vs-inflated proactive catch-rate finding (a repair-loop
   artifact where HDF/PWF-prone robots got re-"caught" every day without the underlying issue ever
   resolving)

Each tab includes rule-based "Key Insight" callouts -- simple threshold/comparison logic evaluated live
against the loaded data (not calling any external AI service).

## Regenerating the underlying data

The dashboard reads exclusively from precomputed CSVs in `data/`. To regenerate them from scratch:

```bash
python explore_data.py            # initial dataset exploration
python cohort_check.py            # Type-vs-wear cohort validation
python validate_thresholds.py     # cross-check against published AI4I generating rules
python rnf_sanity_check.py        # near-miss set integrity check
python extract_near_miss.py       # -> data/near_miss_events.csv
python health_model.py            # health-score model definition + spot checks
python full_eval.py               # -> data/scored_dataset.csv (full-fleet scoring)
python backup_pool_sim.py         # -> data/cost_curve_baseline.csv / _peak.csv (v1: naive)
python shortfall_gated_sim.py     # -> data/shortfall_gated_curve_*.csv (v2: shortfall-gated)
python sensitivity_analysis.py    # holding-cost / swap-cost sensitivity sweeps
python reassignment_sim.py        # v1 dynamic simulation (has the HDF/PWF repair-loop bug, kept for comparison)
python reassignment_sim_v2.py     # v2 dynamic simulation (corrected)
python cost_model_v2.py           # -> data/dynamic_threshold_grid.csv (v3: cumulative-cost correction)
python generate_dashboard_data.py # -> all dashboard-facing CSVs (corrected_cost_curve_*, dynamic_sim_daily_log_*, catch_rate_*)
```

`generate_dashboard_data.py` is the one script the dashboard's data actually depends on; the others are
the analysis trail that produced the validated model it's built on.

## Requirements

Python 3.12. See `requirements.txt`.
