import pandas as pd
import numpy as np
from scipy.optimize import linprog
import warnings

warnings.filterwarnings("ignore")

#load 
df = pd.read_csv("combined_o1_o2_harmonised_acuity.csv")

df = df[[
    "facility_id",
    "group",
    "total_staff",
    "acuity_index",
    "quality_rating",
    "experience_rating",
    "staffing_rating"
]].dropna().copy()

df = df[(df["total_staff"] > 0) & (df["acuity_index"] > 0)].copy()

#weights
df["out_balanced_40_40_20"] = (
    0.4 * df["quality_rating"] +
    0.4 * df["experience_rating"] +
    0.2 * df["staffing_rating"]
)

df["out_quality_60_30_10"] = (
    0.6 * df["quality_rating"] +
    0.3 * df["experience_rating"] +
    0.1 * df["staffing_rating"]
)

df["out_experience_30_60_10"] = (
    0.3 * df["quality_rating"] +
    0.6 * df["experience_rating"] +
    0.1 * df["staffing_rating"]
)

df["out_staffing_40_30_30"] = (
    0.4 * df["quality_rating"] +
    0.3 * df["experience_rating"] +
    0.3 * df["staffing_rating"]
)

#dea function
def run_dea(data, input_cols, output_col, model_name, rts="VRS"):
    X = data[input_cols].values
    Y = data[[output_col]].values

    n = X.shape[0]
    efficiency_scores = []

    for i in range(n):
        c = np.zeros(n + 1)
        c[-1] = -1.0

        A_ub_inputs = np.hstack([X.T, np.zeros((X.shape[1], 1))])
        b_ub_inputs = X[i]

        A_ub_outputs = np.hstack([-Y.T, Y[i].reshape(-1, 1)])
        b_ub_outputs = np.zeros(Y.shape[1])

        A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
        b_ub = np.concatenate([b_ub_inputs, b_ub_outputs])

        bounds = [(0, None)] * n + [(1, None)]

        if rts == "VRS":
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1.0
            b_eq = [1.0]
        elif rts == "NIRS":
            row = np.zeros((1, n + 1))
            row[0, :n] = 1.0
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, 1.0)
            A_eq, b_eq = None, None
        else:  # CRS
            A_eq, b_eq = None, None

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if result.success:
            phi = max(result.x[-1], 1.0)
            eff = np.clip(1.0 / phi, 0.0, 1.0)
        else:
            eff = np.nan

        efficiency_scores.append(eff)

    out = data.copy()
    out["dea_efficiency"] = efficiency_scores
    out["rank"] = out["dea_efficiency"].rank(ascending=False, method="average")
    out["model"] = model_name
    out["rts"] = rts
    out["group_model"] = f"ALL_{model_name}_{rts}"

    return out

#models

models = [
    ("Balanced", "out_balanced_40_40_20"),
    ("Quality", "out_quality_60_30_10"),
    ("Experience", "out_experience_30_60_10"),
    ("Staffing", "out_staffing_40_30_30"),
]

input_cols = ["total_staff", "acuity_index"]

#dea
all_results = []
summary_rows = []

for model_name, output_col in models:
    for rts in ["VRS", "CRS", "NIRS"]:
        res = run_dea(df.copy(), input_cols, output_col, model_name, rts)
        all_results.append(res)

        summary_rows.append({
            "model": model_name,
            "rts": rts,
            "n_facilities": len(res),
            "mean_efficiency": res["dea_efficiency"].mean(),
            "min_efficiency": res["dea_efficiency"].min(),
            "max_efficiency": res["dea_efficiency"].max(),
            "n_efficient": (res["dea_efficiency"] >= 1.0 - 1e-6).sum(),
        })

results = pd.concat(all_results, ignore_index=True)
summary = pd.DataFrame(summary_rows)

#scale efficiency

TOL = 1e-6

vrs_df = (
    results[results["rts"] == "VRS"]
    [["facility_id", "group", "model", "dea_efficiency"]]
    .rename(columns={"dea_efficiency": "eff_vrs"})
)

crs_df = (
    results[results["rts"] == "CRS"]
    [["facility_id", "group", "model", "dea_efficiency"]]
    .rename(columns={"dea_efficiency": "eff_crs"})
)

nirs_df = (
    results[results["rts"] == "NIRS"]
    [["facility_id", "group", "model", "dea_efficiency"]]
    .rename(columns={"dea_efficiency": "eff_nirs"})
)

scale_eff = (
    vrs_df
    .merge(crs_df, on=["facility_id", "group", "model"])
    .merge(nirs_df, on=["facility_id", "group", "model"])
)

scale_eff["scale_efficiency"] = (
    scale_eff["eff_crs"] / scale_eff["eff_vrs"]
).clip(upper=1.0)

def classify_rts(row):
    at_optimal = (row["eff_vrs"] - row["eff_crs"]) < TOL
    nirs_eq_vrs = abs(row["eff_nirs"] - row["eff_vrs"]) < TOL

    if at_optimal:
        return "CRS (optimal scale)"
    elif nirs_eq_vrs:
        return "IRS (scale up)"
    else:
        return "DRS (scale down)"

scale_eff["scale_direction"] = scale_eff.apply(classify_rts, axis=1)

scale_direction_summary = (
    scale_eff.groupby(["model", "scale_direction"])
    .size()
    .reset_index(name="n_facilities")
)

#export

output_path = "dea_pooled_total_staff_final.xlsx"


with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    results.to_excel(writer, sheet_name="All_DEA_Results", index=False)
    scale_eff.to_excel(writer, sheet_name="Scale_Efficiency", index=False)
    scale_direction_summary.to_excel(writer, sheet_name="Scale_Direction_Summary", index=False)

    for gm in results["group_model"].unique():
        temp = results[results["group_model"] == gm].copy()
        sheet_name = gm[:31]
        temp.to_excel(writer, sheet_name=sheet_name, index=False)

print("\nDEA SUMMARY (POOLED ONLY)")
print(summary.to_string(index=False))

print("\nSCALE DIRECTION SUMMARY")
print(scale_direction_summary.to_string(index=False))

print(f"\nSaved Excel file: {output_path}")