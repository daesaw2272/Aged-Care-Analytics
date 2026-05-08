import pandas as pd
import numpy as np

# Use the exact uploaded filename
file_path = "data - RACF.xlsx"

# ============================================================
# O1 DATA
# ============================================================

# Read O1 sheets
o1_staff = pd.read_excel(file_path, sheet_name="data- O1 - Staffing")
o1_quality = pd.read_excel(file_path, sheet_name="data - O1 - Quality Rating ")
o1_residents = pd.read_excel(file_path, sheet_name="data O1 - Residents")

# Clean column names
o1_staff.columns = o1_staff.columns.str.strip()
o1_quality.columns = o1_quality.columns.str.strip()
o1_residents.columns = o1_residents.columns.str.strip()

# Rename quality columns
o1_quality = o1_quality.rename(columns={
    "Quality": "quality_rating",
    "Compliance": "compliance_rating",
    "Res Exp'ce": "experience_rating",
    "Staffing": "staffing_rating",
    "Overall": "overall_rating"
})

# Create consistent facility IDs
o1_staff = o1_staff.rename(columns={"DMU": "facility_id"})
o1_staff["facility_id"] = [f"O1L{i+1}" for i in range(len(o1_staff))]
o1_quality["facility_id"] = o1_staff["facility_id"].values

# Merge staffing and quality
o1 = pd.merge(o1_staff, o1_quality, on="facility_id", how="left")
o1["group"] = "O1"

# ------------------------------------------------------------
# O1 resident acuity from O1 Residents sheet
# ------------------------------------------------------------

# Create facility_id from Location
o1_residents["location_num"] = (
    o1_residents["Location"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

o1_residents["facility_id"] = (
    "O1L" + o1_residents["location_num"].astype("Int64").astype(str)
)

# Extract numeric AN-ACC class
o1_residents["anacc_class"] = (
    o1_residents["Class"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# Keep only valid AN-ACC classes 1–13
o1_residents.loc[
    ~o1_residents["anacc_class"].between(1, 13),
    "anacc_class"
] = np.nan

# Convert residents count to numeric
o1_residents["Residents"] = pd.to_numeric(
    o1_residents["Residents"],
    errors="coerce"
)

# Remove invalid rows before acuity calculation
o1_residents_valid = o1_residents.dropna(
    subset=["facility_id", "anacc_class", "Residents"]
).copy()

# Weighted AN-ACC class
o1_residents_valid["weighted_class"] = (
    o1_residents_valid["anacc_class"] * o1_residents_valid["Residents"]
)

# Aggregate to facility level
o1_acuity = (
    o1_residents_valid
    .groupby("facility_id", as_index=False)
    .agg(
        total_residents=("Residents", "sum"),
        weighted_class_sum=("weighted_class", "sum")
    )
)

# Direct 0–1 acuity index from AN-ACC class 1–13
o1_acuity["acuity_index"] = (
    ((o1_acuity["weighted_class_sum"] / o1_acuity["total_residents"]) - 1) / 12
)

o1 = pd.merge(
    o1,
    o1_acuity[["facility_id", "total_residents", "acuity_index"]],
    on="facility_id",
    how="left"
)

# Create staffing variables for O1
o1["total_staff"] = o1["Admin"] + o1["Life"] + o1["Care"] + o1["Health"]
o1["care_staff"] = o1["Care"] + o1["Health"]
o1["support_staff"] = o1["Admin"] + o1["Life"]

o1["staff_per_resident"] = o1["total_staff"] / o1["total_residents"]
o1["care_ratio"] = o1["care_staff"] / o1["total_staff"]

# Output summary score
o1["quality_score"] = (
    o1["quality_rating"] +
    o1["experience_rating"] +
    o1["staffing_rating"]
) / 3

o1_clean = o1[[
    "facility_id",
    "group",
    "total_residents",
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

# ============================================================
# O2 DATA
# ============================================================

raw_o2 = pd.read_excel(file_path, sheet_name="data O2", header=None)

facility_ids = raw_o2.iloc[0, 3:17].tolist()
location_types = raw_o2.iloc[1, 3:17].tolist()

o2_clean = pd.DataFrame({
    "facility_id": facility_ids,
    "group": "O2",
    "location_type": location_types
})

# AN-ACC classes 1–13
for r in range(4, 17):
    class_num = int(raw_o2.iloc[r, 2])
    o2_clean[f"anacc_{class_num}"] = pd.to_numeric(
        raw_o2.iloc[r, 3:17].values,
        errors="coerce"
    )

o2_clean["total_beds"] = pd.to_numeric(
    raw_o2.iloc[17, 3:17].values,
    errors="coerce"
)

o2_clean["unclassified_respite"] = pd.to_numeric(
    raw_o2.iloc[18, 3:17].values,
    errors="coerce"
)

# Staffing variables
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
        raw_o2.iloc[row_idx, 3:17].values,
        errors="coerce"
    )

# Rating variables
rating_map = {
    "overall_rating": 32,
    "compliance_rating": 33,
    "quality_rating": 34,
    "experience_rating": 35,
    "staffing_rating": 36
}

for var_name, row_idx in rating_map.items():
    o2_clean[var_name] = pd.to_numeric(
        raw_o2.iloc[row_idx, 3:17].values,
        errors="coerce"
    )

# O2 acuity calculation
anacc_cols = [f"anacc_{i}" for i in range(1, 14)]

o2_clean["total_classified_residents"] = o2_clean[anacc_cols].sum(axis=1)
o2_clean["total_residents"] = (
    o2_clean["total_classified_residents"] +
    o2_clean["unclassified_respite"]
)

weights = np.arange(1, 14)

o2_clean["acuity_index"] = (
    (
        o2_clean[anacc_cols].mul(weights, axis=1).sum(axis=1)
        / o2_clean["total_classified_residents"]
    ) - 1
) / 12

# O2 staffing totals
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

o2_clean["staff_per_resident"] = (
    o2_clean["total_staff"] / o2_clean["total_residents"]
)

o2_clean["care_ratio"] = (
    o2_clean["care_staff"] / o2_clean["total_staff"]
)

o2_clean["quality_score"] = (
    o2_clean["quality_rating"] +
    o2_clean["experience_rating"] +
    o2_clean["staffing_rating"]
) / 3

o2_clean = o2_clean[[
    "facility_id",
    "group",
    "total_residents",
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

# ============================================================
# COMBINE O1 + O2
# ============================================================

combined = pd.concat([o1_clean, o2_clean], ignore_index=True)

# Preview
print("O1 clean shape:", o1_clean.shape)
print("O2 clean shape:", o2_clean.shape)
print("Combined shape:", combined.shape)

print("\nO1 preview:")
print(o1_clean.head())

print("\nO2 preview:")
print(o2_clean.head())

print("\nCombined preview:")
print(combined.head())

# Save outputs
o1_clean.to_csv("o1_cleaned_acuity.csv", index=False)
o2_clean.to_csv("o2_cleaned_acuity.csv", index=False)
combined.to_csv("combined_o1_o2_acuity.csv", index=False)

print("\nSaved files:")
print("- o1_cleaned_acuity.csv")
print("- o2_cleaned_acuity.csv")
print("- combined_o1_o2_acuity.csv")