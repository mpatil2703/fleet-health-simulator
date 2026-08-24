import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")
fail_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

near_miss = df[(df["Machine failure"] == 0) & (df[fail_modes].sum(axis=1) >= 1)].copy()
near_miss["tripped_modes"] = near_miss[fail_modes].apply(
    lambda r: "+".join([m for m in fail_modes if r[m] == 1]), axis=1
)

print(f"near-miss rows: {len(near_miss)}")
print("\ntripped mode breakdown:")
print(near_miss["tripped_modes"].value_counts())
print()
cols = ["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "tripped_modes"]
print(near_miss[cols].to_string(index=False))

near_miss.to_csv("data/near_miss_events.csv", index=False)
print("\nsaved -> data/near_miss_events.csv")
