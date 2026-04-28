import pandas as pd
import numpy as np
from scipy.optimize import linprog
import warnings

warnings.filterwarnings("ignore")



df = pd.read_csv("combined_o1_o2_acuity.csv")

df = df[[
    "facility_id",
    "group",
    "care_staff",
    "acuity_index",
    "quality_rating",
    "experience_rating",
    "staffing_rating"
]].dropna().copy()

df = df[(df["care_staff"] > 0) & (df["acuity_index"] > 0)].copy()



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


def run_dea(data, input_cols, output_col, model_name, rts="VRS"):
    X = data[input_cols].values
    Y = data[[output_col]].values

    n = X.shape[0]

    efficiency_scores = []
    peer_rows = []
    target_rows = []

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
        else:
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

        facility_id = data.iloc[i]["facility_id"]
        group = data.iloc[i]["group"]

        if result.success:
            lambdas = result.x[:-1]
            phi = max(result.x[-1], 1.0)
            efficiency = np.clip(1.0 / phi, 0.0, 1.0)

            x_target = lambdas @ X
            y_target = lambdas @ Y

            peer_indices = np.where(lambdas > 1e-6)[0]

            for p in peer_indices:
                peer_rows.append({
                    "facility_id": facility_id,
                    "group": group,
                    "model": model_name,
                    "rts": rts,
                    "peer_facility": data.iloc[p]["facility_id"],
                    "lambda_weight": lambdas[p]
                })

            target = {
                "facility_id": facility_id,
                "group": group,
                "model": model_name,
                "rts": rts,
                "efficiency": efficiency,
                "current_output": Y[i, 0],
                "target_output": y_target[0],
                "output_gap": max(y_target[0] - Y[i, 0], 0),
                "output_gap_pct": (
                    (y_target[0] - Y[i, 0]) / Y[i, 0] * 100
                    if Y[i, 0] != 0 else np.nan
                )
            }

            for j, col in enumerate(input_cols):
                target[f"current_{col}"] = X[i, j]
                target[f"target_{col}"] = x_target[j]
                target[f"{col}_reduction"] = max(X[i, j] - x_target[j], 0)
                target[f"{col}_reduction_pct"] = (
                    (X[i, j] - x_target[j]) / X[i, j] * 100
                    if X[i, j] != 0 else np.nan
                )

            target_rows.append(target)

        else:
            efficiency = np.nan

        efficiency_scores.append(efficiency)

    out = data.copy()
    out["dea_efficiency"] = efficiency_scores
    out["rank"] = out["dea_efficiency"].rank(ascending=False)
    out["model"] = model_name
    out["rts"] = rts

    return out, pd.DataFrame(peer_rows), pd.DataFrame(target_rows)



models = [
    ("Balanced", "out_balanced_40_40_20"),
    ("Quality", "out_quality_60_30_10"),
    ("Experience", "out_experience_30_60_10"),
    ("Staffing", "out_staffing_40_30_30"),
]

input_cols = ["care_staff", "acuity_index"]

all_results = []
all_peers = []
all_targets = []
summary_rows = []

for model_name, output_col in models:
    for rts in ["VRS", "CRS"]:
        res, peers, targets = run_dea(df.copy(), input_cols, output_col, model_name, rts)

        all_results.append(res)
        all_peers.append(peers)
        all_targets.append(targets)

        summary_rows.append({
            "model": model_name,
            "rts": rts,
            "mean_eff": res["dea_efficiency"].mean(),
            "min_eff": res["dea_efficiency"].min(),
            "max_eff": res["dea_efficiency"].max(),
            "n_efficient": (res["dea_efficiency"] >= 1 - 1e-6).sum()
        })

results = pd.concat(all_results)
peers = pd.concat(all_peers)
targets = pd.concat(all_targets)
summary = pd.DataFrame(summary_rows)


inefficient_targets = targets[targets["efficiency"] < 1 - 1e-6].copy()



vrs = results[results["rts"] == "VRS"][["facility_id", "group", "model", "dea_efficiency"]]
vrs = vrs.rename(columns={"dea_efficiency": "eff_vrs"})

crs = results[results["rts"] == "CRS"][["facility_id", "group", "model", "dea_efficiency"]]
crs = crs.rename(columns={"dea_efficiency": "eff_crs"})

scale_eff = vrs.merge(crs, on=["facility_id", "group", "model"])
scale_eff["scale_efficiency"] = scale_eff["eff_crs"] / scale_eff["eff_vrs"]



with pd.ExcelWriter("DEA_carestaff.xlsx") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    results.to_excel(writer, sheet_name="All_Results", index=False)
    peers.to_excel(writer, sheet_name="Benchmark_Peers", index=False)
    targets.to_excel(writer, sheet_name="All_Targets", index=False)
    inefficient_targets.to_excel(writer, sheet_name="Inefficient_Targets", index=False)
    scale_eff.to_excel(writer, sheet_name="Scale_Efficiency", index=False)
