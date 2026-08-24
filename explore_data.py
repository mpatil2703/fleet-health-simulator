import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("data/ai4i2020.csv")

print("=" * 70)
print("SHAPE:", df.shape)
print("=" * 70)

print("\n--- dtypes ---")
print(df.dtypes)

print("\n--- head ---")
print(df.head(10))

print("\n--- missing values ---")
print(df.isna().sum())

print("\n--- duplicate UDI / Product ID check ---")
print("duplicate UDI rows:", df["UDI"].duplicated().sum())
print("duplicate Product ID rows:", df["Product ID"].duplicated().sum())

print("\n--- Type (product quality variant) distribution ---")
print(df["Type"].value_counts())
print(df["Type"].value_counts(normalize=True).round(3))

print("\n--- numeric summary stats ---")
num_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
print(df[num_cols].describe().T)

print("\n--- Machine failure (overall target) balance ---")
print(df["Machine failure"].value_counts())
print("failure rate:", df["Machine failure"].mean().round(4))

print("\n--- individual failure mode flags ---")
fail_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
for col in fail_modes:
    n = df[col].sum()
    print(f"  {col}: {n} occurrences ({n/len(df)*100:.2f}%)")

print("\n--- sum of individual flags vs Machine failure total ---")
print("sum of all flags (rows can have >1):", df[fail_modes].sum().sum())
print("rows with >=1 flag set:", (df[fail_modes].sum(axis=1) >= 1).sum())
print("rows with Machine failure == 1:", df["Machine failure"].sum())

print("\n--- mismatch check: Machine failure=1 but no flag set (ambiguous failures) ---")
mismatch = df[(df["Machine failure"] == 1) & (df[fail_modes].sum(axis=1) == 0)]
print("count:", len(mismatch))

print("\n--- mismatch check: a flag set but Machine failure=0 ---")
mismatch2 = df[(df["Machine failure"] == 0) & (df[fail_modes].sum(axis=1) >= 1)]
print("count:", len(mismatch2))

print("\n--- rows with multiple simultaneous failure modes ---")
multi = df[df[fail_modes].sum(axis=1) > 1]
print("count:", len(multi))
if len(multi) > 0:
    print(multi[["UDI", "Product ID", "Type"] + num_cols + ["Machine failure"] + fail_modes].head(10))

print("\n--- failure rate by Type ---")
print(df.groupby("Type")["Machine failure"].agg(["mean", "count"]))

print("\n--- tool wear range at failure vs no failure ---")
print(df.groupby("Machine failure")["Tool wear [min]"].describe())
