# %% Imports and load
import os
import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/hk389/Downloads/October_1_1.csv")
print(df.shape, df.isnull().sum().sum(), df.duplicated().sum())

# %% Rename columns
df.columns = [
    "sl_no", "region", "state", "sector", "plant_or_company", "station_or_fuel",
    "capacity_mw", "target_annual",
    "oct_program", "oct_actual", "oct_actual_last_year", "oct_pct_program", "oct_pct_last_year",
    "ytd_program", "ytd_actual", "ytd_actual_last_year", "ytd_pct_program", "ytd_pct_last_year",
    "plf_oct_program", "plf_oct_actual", "plf_oct_last_year",
    "plf_ytd_program", "plf_ytd_actual", "plf_ytd_last_year",
]
for c in ["region", "state", "sector", "plant_or_company", "station_or_fuel"]:
    df[c] = df[c].astype(str).str.strip().replace("nan", np.nan)

# %% Keep only real plants (remove subtotal rows)
has_sub = df["station_or_fuel"].notna()
df["company"] = df["plant_or_company"].where(has_sub).bfill().str.upper().str.rstrip(". ")
df["fuel_type"] = df["station_or_fuel"].where(has_sub).bfill().str.title()
plants = df[~has_sub].rename(columns={"plant_or_company": "plant_name"})
plants = plants.drop(columns=["sl_no", "station_or_fuel"]).drop_duplicates(["state", "plant_name"])

# %% Missing values
plants["capacity_mw"] = plants["capacity_mw"].fillna(plants["capacity_mw"].median())
plf_cols = [c for c in plants.columns if c.startswith("plf_")]
plants["plf_reported"] = plants["plf_oct_actual"].notna().astype(int)
for c in plf_cols:
    plants[c] = plants.groupby("fuel_type")[c].transform(lambda x: x.fillna(x.median()))
    plants[c] = plants[c].fillna(plants[c].median())

# %% Fix wrong values
plants = plants[plants["capacity_mw"] > 0]
pairs = {"oct_pct_program": ("oct_actual", "oct_program"),
         "oct_pct_last_year": ("oct_actual", "oct_actual_last_year"),
         "ytd_pct_program": ("ytd_actual", "ytd_program"),
         "ytd_pct_last_year": ("ytd_actual", "ytd_actual_last_year")}
for col, (a, b) in pairs.items():
    plants[col] = np.where(plants[b] > 0, plants[a] / plants[b] * 100, np.nan)

# %% Outliers (cap, don't delete)
for c in pairs:
    q1, q3 = plants[c].quantile([0.25, 0.75])
    plants[c] = plants[c].clip(0, q3 + 1.5 * (q3 - q1))
    plants[c] = plants[c].fillna(plants[c].median())

# %% Final check and save
plants = plants.reset_index(drop=True)
print(plants.shape, plants.isnull().sum().sum(), plants.duplicated().sum())
plants.to_csv("power_cleaned.csv", index=False)
print("Saved in:", os.path.abspath("power_cleaned.csv"))