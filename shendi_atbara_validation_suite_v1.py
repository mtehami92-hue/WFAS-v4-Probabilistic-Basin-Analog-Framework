# ============================================================
# Shendi–Atbara Model Validation Suite v1
# Tests: Sensitivity, Ablation, Weight Robustness, Leave-One-Basin-Out
# Required input in Colab:
#   Shendi_Atbara_Project_Master_Matrix_v4_WFAS_MC.xlsx
# or fallback:
#   Shendi_Atbara_Project_Master_Matrix_v4_Spatial.xlsx
# ============================================================

import os
import numpy as np
import pandas as pd

MASTER_XLSX = "Shendi_Atbara_Project_Master_Matrix_v4_WFAS_MC.xlsx"
FALLBACK_XLSX = "Shendi_Atbara_Project_Master_Matrix_v4_Spatial.xlsx"
OUTDIR = "outputs_validation_suite"
RANDOM_SEED = 42
N_WEIGHT_RUNS = 2000

os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

if os.path.exists(MASTER_XLSX):
    workbook = MASTER_XLSX
elif os.path.exists(FALLBACK_XLSX):
    workbook = FALLBACK_XLSX
else:
    raise FileNotFoundError("Upload the v4 workbook first.")

print(f"Using workbook: {workbook}")
xls = pd.ExcelFile(workbook)
print("Available sheets:", xls.sheet_names)

def safe_read(sheet):
    return pd.read_excel(workbook, sheet_name=sheet) if sheet in xls.sheet_names else pd.DataFrame()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p, eps=1e-6):
    p = np.clip(float(p), eps, 1 - eps)
    return np.log(p / (1 - p))

mc_df = safe_read("MonteCarlo_v4")
wfas_df = safe_read("WFAS_v4")
if wfas_df.empty:
    wfas_df = safe_read("Top_Formation_Analogs_v4")
if wfas_df.empty:
    wfas_df = safe_read("Analog_Formations")

if not mc_df.empty and "Formation" in mc_df.columns:
    baseline = mc_df.copy()
else:
    baseline = pd.DataFrame([
        {"Formation": "Mahmiya Fm", "Analog_support_WFAS": 0.781000, "P_source_mean": 0.633299, "P_reservoir_mean": 0.634015, "P_trap_mean": 0.769471, "Risk_mean": 0.448645, "P_petroleum_mean": 0.757540},
        {"Formation": "Shendi Fm", "Analog_support_WFAS": 0.765667, "P_source_mean": 0.610263, "P_reservoir_mean": 0.750726, "P_trap_mean": 0.592418, "Risk_mean": 0.565108, "P_petroleum_mean": 0.700411},
        {"Formation": "Bagrawiya Fm", "Analog_support_WFAS": 0.712467, "P_source_mean": 0.699719, "P_reservoir_mean": 0.686262, "P_trap_mean": 0.563117, "Risk_mean": 0.532719, "P_petroleum_mean": 0.686355},
    ])

required = ["Formation", "Analog_support_WFAS", "P_source_mean", "P_reservoir_mean", "P_trap_mean", "Risk_mean"]
for c in required:
    if c not in baseline.columns:
        raise ValueError(f"Missing baseline column: {c}")

BASE_WEIGHTS = {"analog": 0.22, "source": 0.22, "reservoir": 0.20, "trap": 0.22, "risk": 0.14}

def predict_petroleum(row, weights=None, multipliers=None):
    weights = BASE_WEIGHTS if weights is None else weights
    multipliers = {} if multipliers is None else multipliers
    analog = np.clip(row["Analog_support_WFAS"] * multipliers.get("analog", 1), 0.01, 0.99)
    ps = np.clip(row["P_source_mean"] * multipliers.get("source", 1), 0.01, 0.99)
    pr = np.clip(row["P_reservoir_mean"] * multipliers.get("reservoir", 1), 0.01, 0.99)
    pt = np.clip(row["P_trap_mean"] * multipliers.get("trap", 1), 0.01, 0.99)
    risk = np.clip(row["Risk_mean"] * multipliers.get("risk", 1), 0.01, 0.99)
    z = (weights["analog"] * logit(analog) + weights["source"] * logit(ps) +
         weights["reservoir"] * logit(pr) + weights["trap"] * logit(pt) -
         weights["risk"] * logit(risk))
    return float(sigmoid(z))

baseline["P_model_recomputed"] = baseline.apply(predict_petroleum, axis=1)

# 1) Sensitivity Analysis
perturbations = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]
drivers = ["analog", "source", "reservoir", "trap", "risk"]
sens = []
for driver in drivers:
    for d in perturbations:
        tmp = baseline.copy()
        tmp["P_test"] = tmp.apply(lambda r: predict_petroleum(r, multipliers={driver: 1+d}), axis=1)
        tmp["Delta_P"] = tmp["P_test"] - tmp["P_model_recomputed"]
        for _, r in tmp.iterrows():
            sens.append({"Driver": driver, "Perturbation": d, "Formation": r["Formation"],
                         "P_baseline": r["P_model_recomputed"], "P_test": r["P_test"],
                         "Delta_P": r["Delta_P"], "Abs_Delta_P": abs(r["Delta_P"])})
sensitivity_df = pd.DataFrame(sens)
sensitivity_summary = sensitivity_df.groupby("Driver").agg(
    Mean_Abs_Delta=("Abs_Delta_P", "mean"), Max_Abs_Delta=("Abs_Delta_P", "max"), Mean_Delta=("Delta_P", "mean")
).reset_index().sort_values("Mean_Abs_Delta", ascending=False)

# 2) Ablation Test: neutralize layer to 0.5
ablation = []
def ablated_prediction(row, layer):
    r = row.copy()
    mapping = {"analog": "Analog_support_WFAS", "source": "P_source_mean", "reservoir": "P_reservoir_mean", "trap": "P_trap_mean", "risk": "Risk_mean"}
    r[mapping[layer]] = 0.5
    return predict_petroleum(r)

for layer in drivers:
    tmp = baseline.copy()
    tmp["P_test"] = tmp.apply(lambda r: ablated_prediction(r, layer), axis=1)
    tmp["Delta_P"] = tmp["P_test"] - tmp["P_model_recomputed"]
    for _, r in tmp.iterrows():
        ablation.append({"Removed_Layer": layer, "Formation": r["Formation"],
                         "P_baseline": r["P_model_recomputed"], "P_test": r["P_test"],
                         "Delta_P": r["Delta_P"], "Abs_Delta_P": abs(r["Delta_P"])})
ablation_df = pd.DataFrame(ablation)
ablation_summary = ablation_df.groupby("Removed_Layer").agg(
    Mean_Abs_Delta=("Abs_Delta_P", "mean"), Max_Abs_Delta=("Abs_Delta_P", "max"), Mean_Delta=("Delta_P", "mean")
).reset_index().sort_values("Mean_Abs_Delta", ascending=False)

# 3) Weight Robustness
robust = []
baseline_rank = baseline.sort_values("P_model_recomputed", ascending=False)["Formation"].tolist()

def perturb_weights(pct):
    vals = {k: v * rng.uniform(1-pct, 1+pct) for k, v in BASE_WEIGHTS.items()}
    s = sum(vals.values())
    return {k: v/s for k, v in vals.items()}

for level in [0.10, 0.20, 0.30]:
    for run in range(N_WEIGHT_RUNS):
        w = perturb_weights(level)
        tmp = baseline.copy()
        tmp["P_test"] = tmp.apply(lambda r: predict_petroleum(r, weights=w), axis=1)
        ranked = tmp.sort_values("P_test", ascending=False)["Formation"].tolist()
        top1_same = ranked[0] == baseline_rank[0]
        top3_same = set(ranked[:3]) == set(baseline_rank[:3])
        for _, r in tmp.iterrows():
            robust.append({"Perturbation_Level": level, "Run": run, "Formation": r["Formation"],
                           "P_baseline": float(baseline.loc[baseline.Formation.eq(r["Formation"]), "P_model_recomputed"].iloc[0]),
                           "P_test": r["P_test"], "Delta_P": r["P_test"] - float(baseline.loc[baseline.Formation.eq(r["Formation"]), "P_model_recomputed"].iloc[0]),
                           "Top1_Same": top1_same, "Top3_Set_Same": top3_same})
robustness_df = pd.DataFrame(robust)
robustness_summary = robustness_df.groupby("Perturbation_Level").agg(
    Mean_Abs_Delta=("Delta_P", lambda x: np.mean(np.abs(x))), P_std=("P_test", "std"),
    Top1_Stability=("Top1_Same", "mean"), Top3_Set_Stability=("Top3_Set_Same", "mean")
).reset_index()

# 4) Leave-One-Basin-Out
lobo = []
analog_table = wfas_df.copy()
# normalize common names
ren = {}
for c in analog_table.columns:
    cl = str(c).lower()
    if "target" in cl and "formation" in cl: ren[c] = "Target_Formation"
    elif "analog" in cl and "basin" in cl: ren[c] = "Analog_Basin"
    elif "wfas" in cl: ren[c] = "WFAS_v4"
    elif "score" in cl and "WFAS_v4" not in ren.values(): ren[c] = "WFAS_v4"
analog_table = analog_table.rename(columns=ren)

if all(c in analog_table.columns for c in ["Target_Formation", "Analog_Basin", "WFAS_v4"]):
    for left_out in sorted(analog_table["Analog_Basin"].dropna().unique()):
        train = analog_table[analog_table["Analog_Basin"] != left_out]
        for formation in baseline["Formation"]:
            base_support = float(baseline.loc[baseline.Formation.eq(formation), "Analog_support_WFAS"].iloc[0])
            subset = train[train["Target_Formation"].eq(formation)]
            new_support = float(subset["WFAS_v4"].max()) if not subset.empty else 0.5
            row = baseline[baseline.Formation.eq(formation)].iloc[0].copy()
            row["Analog_support_WFAS"] = new_support
            p = predict_petroleum(row)
            p0 = float(baseline.loc[baseline.Formation.eq(formation), "P_model_recomputed"].iloc[0])
            lobo.append({"Left_Out_Basin": left_out, "Formation": formation,
                         "Baseline_Analog_Support": base_support, "LOBO_Analog_Support": new_support,
                         "P_baseline": p0, "P_test": p, "Delta_P": p-p0, "Abs_Delta_P": abs(p-p0)})
else:
    for reduction in [0.15, 0.30]:
        for _, row0 in baseline.iterrows():
            row = row0.copy()
            base_support = float(row["Analog_support_WFAS"])
            row["Analog_support_WFAS"] = max(0.5, base_support*(1-reduction))
            p = predict_petroleum(row)
            p0 = float(row0["P_model_recomputed"])
            lobo.append({"Left_Out_Basin": f"Proxy_{int(reduction*100)}pct_analog_loss", "Formation": row["Formation"],
                         "Baseline_Analog_Support": base_support, "LOBO_Analog_Support": row["Analog_support_WFAS"],
                         "P_baseline": p0, "P_test": p, "Delta_P": p-p0, "Abs_Delta_P": abs(p-p0)})

lobo_df = pd.DataFrame(lobo)
lobo_summary = lobo_df.groupby("Left_Out_Basin").agg(
    Mean_Abs_Delta=("Abs_Delta_P", "mean"), Max_Abs_Delta=("Abs_Delta_P", "max"), Mean_Delta=("Delta_P", "mean")
).reset_index().sort_values("Mean_Abs_Delta", ascending=False)

# Save
out_xlsx = os.path.join(OUTDIR, "Shendi_Atbara_Validation_Suite_Results.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    baseline.to_excel(writer, sheet_name="Baseline", index=False)
    sensitivity_df.to_excel(writer, sheet_name="Sensitivity_Detail", index=False)
    sensitivity_summary.to_excel(writer, sheet_name="Sensitivity_Summary", index=False)
    ablation_df.to_excel(writer, sheet_name="Ablation_Detail", index=False)
    ablation_summary.to_excel(writer, sheet_name="Ablation_Summary", index=False)
    robustness_summary.to_excel(writer, sheet_name="Weight_Robustness", index=False)
    robustness_df.sample(min(5000, len(robustness_df)), random_state=RANDOM_SEED).to_excel(writer, sheet_name="Weight_Robust_Sample", index=False)
    lobo_df.to_excel(writer, sheet_name="LOBO_Detail", index=False)
    lobo_summary.to_excel(writer, sheet_name="LOBO_Summary", index=False)

sensitivity_summary.to_csv(os.path.join(OUTDIR, "sensitivity_summary.csv"), index=False)
ablation_summary.to_csv(os.path.join(OUTDIR, "ablation_summary.csv"), index=False)
robustness_summary.to_csv(os.path.join(OUTDIR, "weight_robustness_summary.csv"), index=False)
lobo_summary.to_csv(os.path.join(OUTDIR, "leave_one_basin_out_summary.csv"), index=False)

print("\n=== Baseline ===")
print(baseline[["Formation", "P_model_recomputed"]])
print("\n=== Sensitivity Summary ===")
print(sensitivity_summary)
print("\n=== Ablation Summary ===")
print(ablation_summary)
print("\n=== Weight Robustness Summary ===")
print(robustness_summary)
print("\n=== Leave-One-Basin-Out Summary ===")
print(lobo_summary)
print("\nSaved:", out_xlsx)
