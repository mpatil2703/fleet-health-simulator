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

rnf_rows = df[df["RNF"] == 1].copy()
print(f"total RNF=1 rows: {len(rnf_rows)}")

print("\n--- 1. flag overlap: do any RNF rows also have another mode flag set? ---")
other_modes = ["TWF", "HDF", "PWF", "OSF"]
rnf_rows["other_flags_sum"] = rnf_rows[other_modes].sum(axis=1)
print(rnf_rows[["UDI"] + other_modes + ["other_flags_sum"]].to_string(index=False))
print(f"rows with any other flag also set: {(rnf_rows['other_flags_sum'] > 0).sum()}")

print("\n--- 2. mechanistic re-derivation: would OSF/HDF/PWF/TWF's OWN deterministic")
print("    rule independently fire on these rows, even though the flag says 0? ---")
print("    (catches mislabeling / coincidental threshold crossing, not just the raw flag)")

osf_threshold = {"L": 11000, "M": 12000, "H": 13000}
rnf_rows["twp"] = rnf_rows["torque"] * rnf_rows["wear"]
rnf_rows["osf_would_fire"] = rnf_rows.apply(lambda r: r["twp"] > osf_threshold[r["Type"]], axis=1)

rnf_rows["temp_diff"] = rnf_rows["process_temp"] - rnf_rows["air_temp"]
rnf_rows["hdf_would_fire"] = (rnf_rows["temp_diff"] < 8.6) & (rnf_rows["rpm"] < 1380)

rnf_rows["power_w"] = rnf_rows["torque"] * (rnf_rows["rpm"] * 2 * np.pi / 60)
rnf_rows["pwf_would_fire"] = (rnf_rows["power_w"] < 3500) | (rnf_rows["power_w"] > 9000)

rnf_rows["twf_in_window"] = rnf_rows["wear"].between(200, 240)

check_cols = ["UDI", "Type", "torque", "wear", "twp", "osf_would_fire",
              "temp_diff", "rpm", "hdf_would_fire", "power_w", "pwf_would_fire",
              "twf_in_window"]
print(rnf_rows[check_cols].to_string(index=False))

any_mechanistic_overlap = (
    rnf_rows["osf_would_fire"] | rnf_rows["hdf_would_fire"] | rnf_rows["pwf_would_fire"]
).sum()
print(f"\nrows where a DIFFERENT mechanism's deterministic rule would also fire: {any_mechanistic_overlap}")
print(f"rows sitting in the TWF random-window [200,240] (necessary-not-sufficient): {rnf_rows['twf_in_window'].sum()}")

print("\n--- 3. UDI / Product ID uniqueness check within the RNF set ---")
print("duplicate UDIs:", rnf_rows["UDI"].duplicated().sum())
print("duplicate Product IDs:", rnf_rows["Product ID"].duplicated().sum())

print("\n--- 4. confirm Machine failure label consistency ---")
print("RNF rows where Machine failure == 1 (i.e. RNF alone was enough to fail the unit):")
print(rnf_rows[rnf_rows["failure"] == 1][["UDI", "failure"]].to_string(index=False))
print(f"count: {(rnf_rows['failure'] == 1).sum()} / {len(rnf_rows)}")

print("\n--- SUMMARY ---")
print(f"19 RNF rows total. {(rnf_rows['other_flags_sum'] > 0).sum()} overlap with another RAW flag.")
print(f"{any_mechanistic_overlap} would independently trigger another mode's DETERMINISTIC rule.")
print(f"{(rnf_rows['failure']==1).sum()} actually flipped Machine failure to 1 (the rest, {(rnf_rows['failure']==0).sum()}, are the near-miss set).")
