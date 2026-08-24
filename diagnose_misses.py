import pandas as pd

df = pd.read_csv("data/scored_dataset.csv")
fail_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

missed = df[(df["failure"] == 1) & (df["health_score"] >= 70)]
cols = ["UDI", "Type", "air_temp", "process_temp", "rpm", "torque", "wear",
        "health_score", "risk_OSF", "risk_HDF", "risk_PWF", "risk_TWF",
        "dominant_mode"] + fail_modes
print(missed[cols].to_string(index=False))

print("\nraw flag set for each missed row:")
missed = missed.copy()
missed["raw_flag"] = missed[fail_modes].apply(
    lambda r: "+".join([m for m in fail_modes if r[m] == 1]) or "NONE SET (ambiguous in source data)", axis=1
)
print(missed[["UDI", "raw_flag"]].to_string(index=False))

print("\n--- cross-check against earlier known 'ambiguous failure' rows ---")
print("(Machine failure=1 but no flag set at all, from initial data-quality check)")
ambiguous = df[(df["failure"] == 1) & (df[fail_modes].sum(axis=1) == 0)]
print(f"total ambiguous rows in dataset: {len(ambiguous)}")
print(f"UDIs: {sorted(ambiguous['UDI'].tolist())}")
missed_udis = set(missed["UDI"].tolist())
ambiguous_udis = set(ambiguous["UDI"].tolist())
print(f"overlap between 'missed by model' and 'ambiguous in source data': {len(missed_udis & ambiguous_udis)} / {len(missed_udis)}")
