import pandas as pd
import numpy as np


file_path = "data - RACF.xlsx"   # change if needed


o1_staff = pd.read_excel(file_path, sheet_name="data- O1 - Staffing")
o1_quality = pd.read_excel(file_path, sheet_name="data - O1 - Quality Rating ")


o1_staff.columns = o1_staff.columns.str.strip()
o1_quality.columns = o1_quality.columns.str.strip()


o1_quality = o1_quality.rename(columns={
    "Quality": "quality_rating",
    "Compliance": "compliance_rating",
    "Res Exp'ce": "experience_rating",
    "Staffing": "staffing_rating",
    "Overall": "overall_rating"
})


o1_staff = o1_staff.rename(columns={"DMU": "facility_id"})


o1_staff["facility_id"] = [
    f"O1L{i+1}" for i in range(len(o1_staff))
]


o1_quality["facility_id"] = o1_staff["facility_id"].values


o1 = pd.merge(o1_staff, o1_quality, on="facility_id", how="left")


o1["group"] = "O1"


o1["total_residents"] = (
    o1["AN-ACC - G1"] +
    o1["AN-ACC - G2"] +
    o1["AN-ACC - G3"] +
    o1["AN-ACC - G4"]
)


o1_weights = [1, 2, 3, 4]
o1["acuity_index_raw"] = (
    o1[["AN-ACC - G1", "AN-ACC - G2", "AN-ACC - G3", "AN-ACC - G4"]]
    .mul(o1_weights, axis=1)
    .sum(axis=1)
    / o1["total_residents"]
)


o1["acuity_index_harmonised"] = (o1["acuity_index_raw"] - 1) / (4 - 1)


o1["acuity_index"] = o1["acuity_index_harmonised"]


o1["total_staff"] = o1["Admin"] + o1["Life"] + o1["Care"] + o1["Health"]
o1["care_staff"] = o1["Care"] + o1["Health"]
o1["support_staff"] = o1["Admin"] + o1["Life"]


o1["staff_per_resident"] = o1["total_staff"] / o1["total_residents"]
o1["care_ratio"] = o1["care_staff"] / o1["total_staff"]


o1["quality_score"] = (
    o1["quality_rating"] +
    o1["experience_rating"] +
    o1["staffing_rating"]
) / 3


o1_clean = o1[[
    "facility_id",
    "group",
    "total_residents",
    "acuity_index_raw",
    "acuity_index_harmonised",
    "acuity_index",
    "total_staff",
    "care_staff",
    "support_staff",
    "staff_per_resident",
    "care_ratio",
    "quality_rating",
    "experience_rating",
    "staffing_rating",
    "compliance_rating",
    "overall_rating",
    "quality_score"
]].copy()


raw_o2 = pd.read_excel(file_path, sheet_name="data O2", header=None)

facility_ids = raw_o2.iloc[0, 3:17].tolist()
location_types = raw_o2.iloc[1, 3:17].tolist()

o2_clean = pd.DataFrame({
    "facility_id": facility_ids,
    "group": "O2",
    "location_type": location_types
})

# AN-ACC classes 1-13
for r in range(4, 17):
    class_num = int(raw_o2.iloc[r, 2])
    o2_clean[f"anacc_{class_num}"] = pd.to_numeric(
        raw_o2.iloc[r, 3:17].values, errors="coerce"
    )


o2_clean["total_beds"] = pd.to_numeric(raw_o2.iloc[17, 3:17].values, errors="coerce")
o2_clean["unclassified_respite"] = pd.to_numeric(raw_o2.iloc[18, 3:17].values, errors="coerce")


staff_map = {
    "registered_nurses": 20,
    "enrolled_nurses": 21,
    "agency": 22,
    "nursing_assistants": 23,
    "chefs": 26,
    "allied_health": 27,
    "other_support_staff": 28,
    "backroom_staff": 29
}

for var_name, row_idx in staff_map.items():
    o2_clean[var_name] = pd.to_numeric(
        raw_o2.iloc[row_idx, 3:17].values, errors="coerce"
    )


rating_map = {
    "overall_rating": 32,
    "compliance_rating": 33,
    "quality_rating": 34,
    "experience_rating": 35,
    "staffing_rating": 36
}

for var_name, row_idx in rating_map.items():
    o2_clean[var_name] = pd.to_numeric(
        raw_o2.iloc[row_idx, 3:17].values, errors="coerce"
    )


anacc_cols = [f"anacc_{i}" for i in range(1, 14)]

o2_clean["total_classified_residents"] = o2_clean[anacc_cols].sum(axis=1)
o2_clean["total_residents"] = (
    o2_clean["total_classified_residents"] + o2_clean["unclassified_respite"]
)


weights = np.arange(1, 14)
o2_clean["acuity_index_raw"] = (
    o2_clean[anacc_cols].mul(weights, axis=1).sum(axis=1)
    / o2_clean["total_classified_residents"]
)


o2_clean["acuity_index_harmonised"] = (o2_clean["acuity_index_raw"] - 1) / (13 - 1)


o2_clean["acuity_index"] = o2_clean["acuity_index_harmonised"]


o2_clean["care_staff"] = (
    o2_clean["registered_nurses"] +
    o2_clean["enrolled_nurses"] +
    o2_clean["agency"] +
    o2_clean["nursing_assistants"]
)

o2_clean["support_staff"] = (
    o2_clean["chefs"] +
    o2_clean["allied_health"] +
    o2_clean["other_support_staff"] +
    o2_clean["backroom_staff"]
)

o2_clean["total_staff"] = o2_clean["care_staff"] + o2_clean["support_staff"]


o2_clean["staff_per_resident"] = o2_clean["total_staff"] / o2_clean["total_residents"]
o2_clean["care_ratio"] = o2_clean["care_staff"] / o2_clean["total_staff"]


o2_clean["quality_score"] = (
    o2_clean["quality_rating"] +
    o2_clean["experience_rating"] +
    o2_clean["staffing_rating"]
) / 3


o2_clean = o2_clean[[
    "facility_id",
    "group",
    "total_residents",
    "acuity_index_raw",
    "acuity_index_harmonised",
    "acuity_index",
    "total_staff",
    "care_staff",
    "support_staff",
    "staff_per_resident",
    "care_ratio",
    "quality_rating",
    "experience_rating",
    "staffing_rating",
    "compliance_rating",
    "overall_rating",
    "quality_score"
]].copy()


combined = pd.concat([o1_clean, o2_clean], ignore_index=True)


# Optional log variables
eps = 1e-6
for d in [o1_clean, o2_clean, combined]:
    d["ln_y"] = np.log(d["quality_score"] + eps)
    d["ln_staff"] = np.log(d["total_staff"] + eps)
    d["ln_residents"] = np.log(d["total_residents"] + eps)
    d["ln_acuity"] = np.log(d["acuity_index"] + eps)


print("O1 clean shape:", o1_clean.shape)
print("O2 clean shape:", o2_clean.shape)
print("Combined shape:", combined.shape)

print("\nO1 acuity preview:")
print(o1_clean[["facility_id", "acuity_index_raw", "acuity_index_harmonised"]].head())

print("\nO2 acuity preview:")
print(o2_clean[["facility_id", "acuity_index_raw", "acuity_index_harmonised"]].head())


o1_clean.to_csv("o1_cleaned_harmonised_acuity.csv", index=False)
o2_clean.to_csv("o2_cleaned_harmonised_acuity.csv", index=False)
combined.to_csv("combined_o1_o2_harmonised_acuity.csv", index=False)

print("\nSaved files:")
print("- o1_cleaned_harmonised_acuity.csv")
print("- o2_cleaned_harmonised_acuity.csv")
print("- combined_o1_o2_harmonised_acuity.csv")