import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

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

fail_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("=" * 70)
print("STEP 1 — is wear itself distributed differently by Type?")
print("(if L machines are simply observed at higher wear, the raw failure")
print(" rate gap could just be an intensity artifact, not a cohort effect)")
print("=" * 70)
print(df.groupby("Type")["wear"].describe()[["count", "mean", "50%", "std"]])

print("\n" + "=" * 70)
print("STEP 2 — failure rate by Type WITHIN fixed wear bins")
print("(this is the direct control: same wear band, does Type still matter?)")
print("=" * 70)
bins = [0, 50, 100, 150, 200, 260]
labels = ["0-50", "50-100", "100-150", "150-200", "200-260"]
df["wear_bin"] = pd.cut(df["wear"], bins=bins, labels=labels, include_lowest=True)

pivot = df.groupby(["wear_bin", "Type"], observed=True)["failure"].agg(["mean", "count"])
print(pivot)

print("\n--- reshaped: failure rate by Type x wear_bin ---")
rate_table = df.groupby(["wear_bin", "Type"], observed=True)["failure"].mean().unstack()
count_table = df.groupby(["wear_bin", "Type"], observed=True)["failure"].count().unstack()
print("\nfailure rate:")
print(rate_table.round(4))
print("\nsample count:")
print(count_table)

print("\n" + "=" * 70)
print("STEP 3 — logistic regression: failure ~ wear + Type (+ wear^2)")
print("If Type coefficients stay significant after wear is in the model,")
print("the gap survives controlling for usage intensity.")
print("=" * 70)
df["wear_sq"] = df["wear"] ** 2
model = smf.logit("failure ~ wear + wear_sq + C(Type, Treatment(reference='H'))", data=df).fit(disp=0)
print(model.summary())

print("\n" + "=" * 70)
print("STEP 4 — same regression but also controlling for torque")
print("(OSF in this dataset is driven by a torque x wear interaction, so")
print(" torque is a relevant confound alongside wear)")
print("=" * 70)
model2 = smf.logit(
    "failure ~ wear + wear_sq + torque + C(Type, Treatment(reference='H'))", data=df
).fit(disp=0)
print(model2.summary())

print("\n" + "=" * 70)
print("STEP 5 — decompose by failure mode: is the Type gap concentrated in OSF?")
print("(OSF = overstrain failure, mechanically tied to torque x wear)")
print("=" * 70)
for mode in fail_modes:
    print(f"\n--- {mode} rate by Type ---")
    print(df.groupby("Type")[mode].agg(["mean", "sum", "count"]))

print("\n--- OSF specifically: torque x wear product by Type, among OSF=1 rows ---")
df["torque_wear_product"] = df["torque"] * df["wear"]
osf_rows = df[df["OSF"] == 1]
print(osf_rows.groupby("Type")["torque_wear_product"].describe()[["count", "mean", "min", "max"]])

print("\n--- OSF rate by Type WITHIN torque_wear_product bins (direct mechanism control) ---")
df["twp_bin"] = pd.cut(
    df["torque_wear_product"],
    bins=[0, 4000, 6000, 8000, 10000, 12000, 14000, 20000],
    include_lowest=True,
)
osf_pivot = df.groupby(["twp_bin", "Type"], observed=True)["OSF"].agg(["mean", "count"])
print(osf_pivot)
