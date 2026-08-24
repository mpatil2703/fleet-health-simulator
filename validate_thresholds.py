import pandas as pd
import numpy as np

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

print("=" * 70)
print("OSF — published rule: torque x wear > {L:11000, M:12000, H:13000} minNm")
print("=" * 70)
osf_threshold = {"L": 11000, "M": 12000, "H": 13000}
df["twp"] = df["torque"] * df["wear"]
df["osf_predicted"] = df.apply(lambda r: 1 if r["twp"] > osf_threshold[r["Type"]] else 0, axis=1)
match = (df["osf_predicted"] == df["OSF"]).sum()
print(f"rows matching published rule exactly: {match}/{len(df)} ({match/len(df)*100:.2f}%)")
mismatch = df[df["osf_predicted"] != df["OSF"]]
print(f"mismatches: {len(mismatch)}")
if len(mismatch) > 0:
    print(mismatch[["UDI", "Type", "torque", "wear", "twp", "OSF", "osf_predicted"]].to_string(index=False))

print("\n" + "=" * 70)
print("HDF — published rule: (process_temp - air_temp) < 8.6K AND rpm < 1380")
print("=" * 70)
df["temp_diff"] = df["process_temp"] - df["air_temp"]
df["hdf_predicted"] = ((df["temp_diff"] < 8.6) & (df["rpm"] < 1380)).astype(int)
match = (df["hdf_predicted"] == df["HDF"]).sum()
print(f"rows matching published rule exactly: {match}/{len(df)} ({match/len(df)*100:.2f}%)")
mismatch = df[df["hdf_predicted"] != df["HDF"]]
print(f"mismatches: {len(mismatch)}")
if len(mismatch) > 0:
    print(mismatch[["UDI", "Type", "air_temp", "process_temp", "temp_diff", "rpm", "HDF", "hdf_predicted"]].head(20).to_string(index=False))

print("\n" + "=" * 70)
print("PWF — published rule: power = torque x rpm[rad/s] outside [3500, 9000] W")
print("=" * 70)
df["power_w"] = df["torque"] * (df["rpm"] * 2 * np.pi / 60)
df["pwf_predicted"] = ((df["power_w"] < 3500) | (df["power_w"] > 9000)).astype(int)
match = (df["pwf_predicted"] == df["PWF"]).sum()
print(f"rows matching published rule exactly: {match}/{len(df)} ({match/len(df)*100:.2f}%)")
mismatch = df[df["pwf_predicted"] != df["PWF"]]
print(f"mismatches: {len(mismatch)}")
if len(mismatch) > 0:
    print(mismatch[["UDI", "Type", "torque", "rpm", "power_w", "PWF", "pwf_predicted"]].head(20).to_string(index=False))

print("\n" + "=" * 70)
print("TWF — published rule: tool reaches a randomly-assigned wear-out time")
print("between 200-240 min, then replaced OR fails (probabilistic, not")
print("deterministic from wear alone -- can only check the NECESSARY condition:")
print("TWF=1 rows should have wear in [200, 240]")
print("=" * 70)
twf_rows = df[df["TWF"] == 1]
print(f"TWF=1 rows: {len(twf_rows)}")
print(twf_rows["wear"].describe()[["min", "max", "mean"]])
in_range = twf_rows["wear"].between(200, 240).sum()
print(f"TWF=1 rows with wear in [200,240]: {in_range}/{len(twf_rows)}")
out_of_range = twf_rows[~twf_rows["wear"].between(200, 240)]
if len(out_of_range) > 0:
    print("\nout-of-range TWF rows:")
    print(out_of_range[["UDI", "Type", "wear", "TWF"]].to_string(index=False))

print("\n--- reverse check: of ALL rows with wear in [200,240], how many are TWF=1? ---")
in_wear_band = df[df["wear"].between(200, 240)]
print(f"rows with wear in [200,240]: {len(in_wear_band)}")
print(f"of those, TWF=1: {in_wear_band['TWF'].sum()}  ({in_wear_band['TWF'].mean()*100:.1f}%)")
print("(published rule implies ~57.5% odds once a machine reaches its randomly")
print(" assigned wear-out point in this window -- not all 200-240 rows are AT")
print(" their assigned point, so this won't hit 57.5% exactly, just a sanity check)")

print("\n" + "=" * 70)
print("RNF — published rule: flat 0.1% chance per row, independent of parameters")
print("=" * 70)
print(f"actual RNF count: {df['RNF'].sum()} / {len(df)} = {df['RNF'].mean()*100:.3f}%")
print("expected under 0.1% flat rate: ~10 occurrences")
print(f"observed: {df['RNF'].sum()} -- {'consistent with' if df['RNF'].sum() < 25 else 'notably above'} pure random noise at this sample size")
