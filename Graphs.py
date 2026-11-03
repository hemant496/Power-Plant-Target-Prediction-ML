# EDA - Monthly Power Generation October 2023
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("eda_plots", exist_ok=True)
df = pd.read_csv("C:/Users/hk389/Downloads/power_cleaned.csv")
colors = ["#c44e52", "#4c82b8"]

def save(name):
    plt.tight_layout()
    plt.savefig("eda_plots/" + name + ".png", dpi=200)
    plt.show()
    plt.close()

# only plants that had a target, then make the target column
df = df[df["oct_program"] > 0].copy()
df["met_target"] = (df["oct_actual"] >= df["oct_program"]).astype(int)
df["capacity_group"] = pd.cut(df["capacity_mw"], [0, 100, 500, 1000, np.inf],
                              labels=["Small", "Medium", "Large", "Very Large"])
df["energy_type"] = np.where(df["fuel_type"] == "Hydro", "Renewable (Hydro)", "Non-renewable")
print(df.shape)
print(df["met_target"].value_counts())

# 1. how many plants met the target
sns.countplot(x="met_target", hue="met_target", data=df, palette=colors, legend=False)
plt.xticks([0, 1], ["Missed (0)", "Met (1)"])
plt.title("Target Achievement Distribution")
plt.xlabel("")
plt.ylabel("Number of plants")
save("fig01_target")

# 2. capacity histogram
sns.histplot(data=df, x="capacity_mw", hue="met_target", bins=30, palette=colors)
plt.title("Capacity Distribution by Target Status")
plt.xlabel("Capacity (MW)")
save("fig02_capacity_hist")

# 3-5, 8, 9. count plots for the categorical columns
cat_plots = [("capacity_group", "Target Status by Capacity Group", "fig03_capacity_group"),
             ("fuel_type", "Target Status by Fuel Type", "fig04_fuel_type"),
             ("region", "Target Status by Region", "fig05_region"),
             ("sector", "Target Status by Sector", "fig08_sector"),
             ("energy_type", "Renewable vs Non-renewable Plants", "fig09_energy_type")]
for col, title, name in cat_plots:
    sns.countplot(x=col, hue="met_target", data=df, palette=colors)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Number of plants")
    if col == "region":
        plt.xticks(rotation=20)
    save(name)

# 6-7. box plots
box_plots = [("oct_actual", "October Generation by Target Status", "Generation (MU)", "fig06_generation_box"),
             ("plf_oct_actual", "Plant Load Factor by Target Status", "PLF (%)", "fig07_plf_box")]
for col, title, ylabel, name in box_plots:
    sns.boxplot(x="met_target", y=col, hue="met_target", data=df, palette=colors, legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    save(name)

# 10. top 10 states
df.groupby("state")["oct_actual"].sum().nlargest(10).sort_values().plot(kind="barh", color="#4c82b8")
plt.title("Top 10 States by October 2023 Generation")
plt.xlabel("Generation (MU)")
save("fig10_top_states")

# 11. generation by fuel
df.groupby("fuel_type")["oct_actual"].sum().sort_values().plot(kind="barh", color="#55a868")
plt.title("October 2023 Generation by Fuel Type")
plt.xlabel("Generation (MU)")
save("fig11_fuel_generation")

# 12. this year vs last year
g = df.groupby("region")[["oct_actual_last_year", "oct_actual"]].sum()
g.columns = ["Oct 2022", "Oct 2023"]
g.plot(kind="bar", color=["#8172b2", "#4c82b8"])
plt.xticks(rotation=20)
plt.title("Regional Generation: Oct 2023 vs Oct 2022")
plt.ylabel("Generation (MU)")
save("fig12_year_comparison")

# 13. capacity vs generation
sns.scatterplot(data=df, x="capacity_mw", y="oct_actual", hue="fuel_type", alpha=0.7)
plt.title("Capacity vs October Generation")
plt.xlabel("Capacity (MW)")
plt.ylabel("Generation (MU)")
save("fig13_scatter")

# 14. correlation heatmap
num_cols = ["capacity_mw", "oct_program", "oct_actual", "oct_actual_last_year",
            "ytd_actual", "plf_oct_actual", "met_target"]
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation Heatmap")
save("fig14_correlation")

# target rate for each group (useful numbers for the report)
for col in ["fuel_type", "region", "sector", "capacity_group"]:
    print(df.groupby(col, observed=True)["met_target"].mean().round(3))

df.to_csv("power_eda.csv", index=False)