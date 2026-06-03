# ============================================================
# Shendi–Atbara AI-Guided Basin Model v4
# WFAS + Spatial Modifiers + Log-Odds Monte Carlo
# ------------------------------------------------------------
# Required inputs in the same Colab folder:
#   1) Shendi_Atbara_Project_Master_Matrix_v4_Spatial.xlsx
#      OR Shendi_Atbara_Project_Master_Matrix_v2_Muglad_fixed.xlsx
#   2) Spatial_Features_60m.csv  (optional if sheets already in Excel)
#
# Outputs:
#   outputs_v4/WFAS_Formation_Analogs_v4.csv
#   outputs_v4/Monte_Carlo_v4_LogOdds.csv
#   outputs_v4/Risk_Decomposition_v4.csv
#   Shendi_Atbara_Project_Master_Matrix_v4_WFAS_MC.xlsx
# ============================================================

import os
import math
import numpy as np
import pandas as pd

# -----------------------------
# User settings
# -----------------------------
MASTER_XLSX = "Shendi_Atbara_Project_Master_Matrix_v4_Spatial.xlsx"
FALLBACK_MASTER_XLSX = "Shendi_Atbara_Project_Master_Matrix_v2_Muglad_fixed.xlsx"
SPATIAL_CSV = "Spatial_Features_60m.csv"
OUTPUT_DIR = "outputs_v4"
OUTPUT_XLSX = "Shendi_Atbara_Project_Master_Matrix_v4_WFAS_MC.xlsx"
N_MC = 10000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# -----------------------------
# Numerical helpers
# -----------------------------
def clip01(x):
    return np.clip(x, 1e-4, 1 - 1e-4)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = clip01(p)
    return np.log(p / (1 - p))


def beta_sample_from_mean(mean, cv=0.28, size=1):
    """Sample bounded probability around a mean using Beta distribution.
    cv is conservative for frontier basins; automatically clipped."""
    mean = float(np.nan_to_num(mean, nan=0.5))
    mean = float(clip01(mean))
    var = (cv * mean) ** 2
    max_var = mean * (1 - mean) * 0.95
    var = min(var, max_var)
    if var <= 1e-8:
        return np.full(size, mean)
    common = mean * (1 - mean) / var - 1
    a = max(mean * common, 0.5)
    b = max((1 - mean) * common, 0.5)
    return np.random.beta(a, b, size=size)


def norm01(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (s - mn) / (mx - mn)

# -----------------------------
# Load workbook
# -----------------------------
if os.path.exists(MASTER_XLSX):
    master_path = MASTER_XLSX
elif os.path.exists(FALLBACK_MASTER_XLSX):
    master_path = FALLBACK_MASTER_XLSX
else:
    raise FileNotFoundError("No master Excel file found. Place v4 spatial or v2 Muglad workbook in Colab folder.")

print(f"Using master workbook: {master_path}")
xl = pd.ExcelFile(master_path)
sheets = {s: pd.read_excel(master_path, sheet_name=s) for s in xl.sheet_names}

required = ["Formation_Features", "Analog_Formations", "Risk_Factors"]
for r in required:
    if r not in sheets:
        raise ValueError(f"Required sheet missing: {r}")

formation_df = sheets["Formation_Features"].copy()
analog_df = sheets["Analog_Formations"].copy()
risk_df = sheets["Risk_Factors"].copy()

# -----------------------------
# Load spatial inputs
# -----------------------------
spatial_inputs = None
formation_spatial = None

if "Spatial_Model_Inputs" in sheets:
    spatial_inputs = sheets["Spatial_Model_Inputs"].copy()
if "Formation_Spatial_Modifiers" in sheets:
    formation_spatial = sheets["Formation_Spatial_Modifiers"].copy()

# If Excel lacks spatial sheets, load Spatial_Features_60m.csv and build summary.
if spatial_inputs is None and os.path.exists(SPATIAL_CSV):
    spatial_df = pd.read_csv(SPATIAL_CSV)
    def get_mean(feature):
        row = spatial_df[spatial_df["Feature"].eq(feature)]
        col = f"{feature}_mean"
        if row.empty or col not in row.columns:
            return np.nan
        return float(row.iloc[0][col])
    feox_mean = get_mean("FeOx_B2_B1")
    clay_mean = get_mean("Clay_B4_B6")
    ferrous_mean = get_mean("Ferrous_B4_B3N")
    gravity_low_mean = get_mean("Gravity_Low_Score")
    gravity_gradient_mean = get_mean("Gravity_Gradient_Score")
    spatial_inputs = pd.DataFrame([{
        "AOI": "Shendi_Atbara_Extended_AOI_32.5_15.5_35.5_18.5",
        "Resolution_m": 60,
        "FeOx_mean": feox_mean,
        "Clay_mean": clay_mean,
        "Ferrous_mean": ferrous_mean,
        "Gravity_low_score_mean": gravity_low_mean,
        "Gravity_gradient_score_mean": gravity_gradient_mean,
        "RemoteSensing_Confidence": 0.75,
        "Gravity_Confidence": 0.65,
    }])

if spatial_inputs is None:
    print("Warning: no spatial inputs found. v4 will run without NASA/TOPEX modifiers.")
    spatial_inputs = pd.DataFrame([{
        "FeOx_mean": np.nan,
        "Clay_mean": np.nan,
        "Ferrous_mean": np.nan,
        "Gravity_low_score_mean": np.nan,
        "Gravity_gradient_score_mean": np.nan,
        "RemoteSensing_Confidence": 0.0,
        "Gravity_Confidence": 0.0,
    }])

if formation_spatial is None:
    base = spatial_inputs.iloc[0]
    feox = base.get("FeOx_mean", np.nan)
    clay = base.get("Clay_mean", np.nan)
    fer = base.get("Ferrous_mean", np.nan)
    gl = base.get("Gravity_low_score_mean", np.nan)
    gg = base.get("Gravity_gradient_score_mean", np.nan)
    formation_spatial = pd.DataFrame([
        {"Formation": "Bagrawiya Fm", "FeOx_modifier": feox*0.85 if pd.notna(feox) else np.nan,
         "Clay_modifier": clay*1.05 if pd.notna(clay) else np.nan,
         "Ferrous_modifier": fer, "Gravity_low_modifier": gl*0.95 if pd.notna(gl) else np.nan,
         "Gravity_gradient_modifier": gg*1.10 if pd.notna(gg) else np.nan},
        {"Formation": "Mahmiya Fm", "FeOx_modifier": feox*0.70 if pd.notna(feox) else np.nan,
         "Clay_modifier": clay*1.20 if pd.notna(clay) else np.nan,
         "Ferrous_modifier": fer*0.90 if pd.notna(fer) else np.nan, "Gravity_low_modifier": gl*1.15 if pd.notna(gl) else np.nan,
         "Gravity_gradient_modifier": gg*0.95 if pd.notna(gg) else np.nan},
        {"Formation": "Shendi Fm", "FeOx_modifier": feox*1.20 if pd.notna(feox) else np.nan,
         "Clay_modifier": clay*0.85 if pd.notna(clay) else np.nan,
         "Ferrous_modifier": fer*1.10 if pd.notna(fer) else np.nan, "Gravity_low_modifier": gl*0.85 if pd.notna(gl) else np.nan,
         "Gravity_gradient_modifier": gg},
    ])

# -----------------------------
# 1) WFAS: Weighted Formation Analog Score
# -----------------------------
# Weights are explicit and editable; calibrated later using Blind_Calibration_Set.
WFAS_WEIGHTS = {
    "Age_Match_0_1": 0.18,
    "Env_Match_0_1": 0.18,
    "Lithology_Match_0_1": 0.16,
    "SSPI_Match_0_1": 0.14,
    "PPI_LCM_Match_0_1": 0.10,
    "Tectonic_Match_0_1": 0.09,
    "Data_Quality_0_1": 0.08,
}
# Remaining 0.07 reserved for spatial support added below.
SPATIAL_WEIGHT = 0.07

analog = analog_df.copy()
for col in WFAS_WEIGHTS:
    if col not in analog.columns:
        analog[col] = 0.5
    analog[col] = pd.to_numeric(analog[col], errors="coerce").fillna(0.5).clip(0, 1)

# Spatial support by target formation: gravity lows support Mahmiya/source; FeOx supports Shendi surface signature but adds diagenetic caution.
fs = formation_spatial.copy()
sp_cols = ["FeOx_modifier", "Clay_modifier", "Gravity_low_modifier", "Gravity_gradient_modifier"]
for col in sp_cols:
    if col not in fs.columns:
        fs[col] = np.nan
    fs[col] = pd.to_numeric(fs[col], errors="coerce")

# Normalize modifiers into [0,1] across formations for score use.
fs["FeOx_norm"] = norm01(fs["FeOx_modifier"]).fillna(0.5)
fs["Clay_norm"] = norm01(fs["Clay_modifier"]).fillna(0.5)
fs["GravityLow_norm"] = norm01(fs["Gravity_low_modifier"]).fillna(0.5)
fs["GravityGradient_norm"] = norm01(fs["Gravity_gradient_modifier"]).fillna(0.5)
fs["Spatial_Support_0_1"] = (
    0.30*fs["FeOx_norm"] + 0.25*fs["Clay_norm"] + 0.35*fs["GravityLow_norm"] + 0.10*(1 - fs["GravityGradient_norm"])
).clip(0,1)

analog = analog.merge(fs[["Formation", "Spatial_Support_0_1"]], how="left", left_on="Target_Formation", right_on="Formation")
analog["Spatial_Support_0_1"] = analog["Spatial_Support_0_1"].fillna(0.5)

analog["WFAS_v4"] = 0.0
for col, w in WFAS_WEIGHTS.items():
    analog["WFAS_v4"] += w * analog[col]
analog["WFAS_v4"] += SPATIAL_WEIGHT * analog["Spatial_Support_0_1"]
analog["WFAS_v4"] = analog["WFAS_v4"].clip(0,1)
analog = analog.sort_values("WFAS_v4", ascending=False)

analog.to_csv(os.path.join(OUTPUT_DIR, "WFAS_Formation_Analogs_v4.csv"), index=False)

# -----------------------------
# 2) Risk decomposition with spatial modifiers
# -----------------------------
risk = risk_df.copy()
for c in ["Seal_Risk_0_1", "Timing_Risk_0_1", "Leakage_Risk_0_1", "Data_Gap_Risk_0_1", "Fault_Risk_0_1"]:
    if c not in risk.columns:
        risk[c] = 0.5
    risk[c] = pd.to_numeric(risk[c], errors="coerce").fillna(0.5).clip(0,1)

risk = risk.merge(fs[["Formation", "GravityGradient_norm", "GravityLow_norm", "Clay_norm", "FeOx_norm"]], on="Formation", how="left")
for c in ["GravityGradient_norm", "GravityLow_norm", "Clay_norm", "FeOx_norm"]:
    risk[c] = risk[c].fillna(0.5)

# Spatially adjusted risks: gradient increases fault/leakage risk; clay reduces seal risk; gravity low slightly reduces timing/source risk.
risk["Seal_Risk_v4"] = (risk["Seal_Risk_0_1"] - 0.08*risk["Clay_norm"] + 0.03*risk["FeOx_norm"]).clip(0,1)
risk["Timing_Risk_v4"] = (risk["Timing_Risk_0_1"] - 0.06*risk["GravityLow_norm"]).clip(0,1)
risk["Leakage_Risk_v4"] = (risk["Leakage_Risk_0_1"] + 0.10*risk["GravityGradient_norm"]).clip(0,1)
risk["Data_Gap_Risk_v4"] = risk["Data_Gap_Risk_0_1"].clip(0,1)
risk["Fault_Risk_v4"] = (risk["Fault_Risk_0_1"] + 0.12*risk["GravityGradient_norm"]).clip(0,1)

risk_weights = {
    "Seal_Risk_v4": 0.25,
    "Timing_Risk_v4": 0.20,
    "Leakage_Risk_v4": 0.25,
    "Data_Gap_Risk_v4": 0.15,
    "Fault_Risk_v4": 0.15,
}
risk["Overall_Risk_v4"] = sum(risk[c]*w for c,w in risk_weights.items()).clip(0,1)
risk.to_csv(os.path.join(OUTPUT_DIR, "Risk_Decomposition_v4.csv"), index=False)

# -----------------------------
# 3) Monte Carlo v4 using weighted log-odds integration
# -----------------------------
target_forms = formation_df[formation_df["Basin"].astype(str).str.contains("Shendi", case=False, na=False)].copy()
if target_forms.empty:
    # fallback to formations with risk entries
    target_forms = formation_df[formation_df["Formation"].isin(risk["Formation"].unique())].copy()

# normalize source helpers from formation features
for col in ["PPI_0_1", "LCM_0_1", "SSPI_0_1", "RSSF_0_1", "Sorting_0_1", "Porosity_frac", "TOC_pct", "Fe_Oxide_0_1"]:
    if col not in target_forms.columns:
        target_forms[col] = np.nan
    target_forms[col] = pd.to_numeric(target_forms[col], errors="coerce")

mc_rows = []
mc_samples_all = []

# contribution weights in log-odds space
LOGIT_WEIGHTS = {
    "source": 0.35,
    "reservoir": 0.30,
    "trap": 0.25,
    "analog": 0.15,
    "risk": 0.40,
}

for _, frow in target_forms.iterrows():
    formation = frow["Formation"]
    rrow = risk[risk["Formation"].eq(formation)]
    if rrow.empty:
        overall_risk = 0.50
        seal_risk = timing_risk = leakage_risk = data_risk = fault_risk = 0.50
    else:
        rr = rrow.iloc[0]
        overall_risk = float(rr.get("Overall_Risk_v4", 0.5))
        seal_risk = float(rr.get("Seal_Risk_v4", 0.5))
        timing_risk = float(rr.get("Timing_Risk_v4", 0.5))
        leakage_risk = float(rr.get("Leakage_Risk_v4", 0.5))
        data_risk = float(rr.get("Data_Gap_Risk_v4", 0.5))
        fault_risk = float(rr.get("Fault_Risk_v4", 0.5))

    # Formation analog support = best analog score for target formation.
    arows = analog[analog["Target_Formation"].eq(formation)]
    analog_support = float(arows["WFAS_v4"].max()) if not arows.empty else 0.5

    # Priors from geological features.
    ppi = float(frow.get("PPI_0_1", np.nan)) if pd.notna(frow.get("PPI_0_1", np.nan)) else 0.55
    lcm = float(frow.get("LCM_0_1", np.nan)) if pd.notna(frow.get("LCM_0_1", np.nan)) else 0.55
    sspi = float(frow.get("SSPI_0_1", np.nan)) if pd.notna(frow.get("SSPI_0_1", np.nan)) else 0.55
    rssf = float(frow.get("RSSF_0_1", np.nan)) if pd.notna(frow.get("RSSF_0_1", np.nan)) else 0.55
    sorting = float(frow.get("Sorting_0_1", np.nan)) if pd.notna(frow.get("Sorting_0_1", np.nan)) else 0.55
    poro = float(frow.get("Porosity_frac", np.nan)) if pd.notna(frow.get("Porosity_frac", np.nan)) else 0.15
    toc = float(frow.get("TOC_pct", np.nan)) if pd.notna(frow.get("TOC_pct", np.nan)) else 1.0
    feox = float(frow.get("Fe_Oxide_0_1", np.nan)) if pd.notna(frow.get("Fe_Oxide_0_1", np.nan)) else 0.5

    # Spatial row.
    fsrow = fs[fs["Formation"].eq(formation)]
    if fsrow.empty:
        spatial_source = spatial_res = spatial_trap = 0.5
    else:
        ff = fsrow.iloc[0]
        spatial_source = float(0.55*ff["GravityLow_norm"] + 0.45*ff["Clay_norm"])
        spatial_res = float(0.45*ff["FeOx_norm"] + 0.35*(1-ff["Clay_norm"]) + 0.20*sspi)
        spatial_trap = float(0.50*(1-ff["GravityGradient_norm"]) + 0.50*ff["GravityLow_norm"])

    # Mean probabilities before MC.
    # Source: boosted by lacustrine/clay/gravity low; TOC is weak until lab data exists.
    p_source_mu = clip01(0.20 + 0.22*lcm + 0.18*ppi + 0.16*spatial_source + 0.12*min(toc/3.0,1) + 0.12*analog_support)
    # Reservoir: boosted by sorting/porosity/SSPI and Shendi-like quartzose maturity; excess clay penalizes.
    p_res_mu = clip01(0.18 + 0.20*sorting + 0.16*sspi + 0.12*rssf + 0.18*min(poro/0.25,1) + 0.16*spatial_res)
    # Trap: boosted by analog/depocenter and penalized by structural gradient through risk in final logit.
    p_trap_mu = clip01(0.22 + 0.20*analog_support + 0.18*spatial_trap + 0.16*(1-fault_risk) + 0.12*(1-leakage_risk) + 0.12*(1-seal_risk))

    # Sample uncertainties conservatively.
    p_source = beta_sample_from_mean(p_source_mu, cv=0.30, size=N_MC)
    p_res = beta_sample_from_mean(p_res_mu, cv=0.28, size=N_MC)
    p_trap = beta_sample_from_mean(p_trap_mu, cv=0.30, size=N_MC)
    p_analog = beta_sample_from_mean(analog_support, cv=0.18, size=N_MC)
    p_risk = beta_sample_from_mean(overall_risk, cv=0.25, size=N_MC)

    # Log-odds evidence integration: avoids product collapse.
    score = (
        LOGIT_WEIGHTS["source"]*logit(p_source) +
        LOGIT_WEIGHTS["reservoir"]*logit(p_res) +
        LOGIT_WEIGHTS["trap"]*logit(p_trap) +
        LOGIT_WEIGHTS["analog"]*logit(p_analog) -
        LOGIT_WEIGHTS["risk"]*logit(p_risk)
    )
    p_petroleum = sigmoid(score)

    mc_rows.append({
        "Formation": formation,
        "Analog_support_WFAS": analog_support,
        "P_source_mean": float(np.mean(p_source)),
        "P_reservoir_mean": float(np.mean(p_res)),
        "P_trap_mean": float(np.mean(p_trap)),
        "Risk_mean": float(np.mean(p_risk)),
        "One_minus_Risk_mean": float(1-np.mean(p_risk)),
        "P_petroleum_mean": float(np.mean(p_petroleum)),
        "P_petroleum_p025": float(np.percentile(p_petroleum, 2.5)),
        "P_petroleum_p500": float(np.percentile(p_petroleum, 50)),
        "P_petroleum_p975": float(np.percentile(p_petroleum, 97.5)),
        "P_petroleum_gt_0_3": float(np.mean(p_petroleum > 0.3)),
        "P_petroleum_gt_0_5": float(np.mean(p_petroleum > 0.5)),
        "Seal_Risk_v4": seal_risk,
        "Timing_Risk_v4": timing_risk,
        "Leakage_Risk_v4": leakage_risk,
        "Data_Gap_Risk_v4": data_risk,
        "Fault_Risk_v4": fault_risk,
        "Method": "Log-odds Monte Carlo v4 with WFAS + ASTER/TOPEX spatial modifiers",
    })

mc_df = pd.DataFrame(mc_rows).sort_values("P_petroleum_mean", ascending=False)
mc_df.to_csv(os.path.join(OUTPUT_DIR, "Monte_Carlo_v4_LogOdds.csv"), index=False)

# -----------------------------
# 4) Write updated workbook with new sheets
# -----------------------------
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    for s, df in sheets.items():
        # Excel sheet max length = 31
        df.to_excel(writer, sheet_name=s[:31], index=False)
    if "Spatial_Model_Inputs" not in sheets:
        spatial_inputs.to_excel(writer, sheet_name="Spatial_Model_Inputs", index=False)
    if "Formation_Spatial_Modifiers" not in sheets:
        formation_spatial.to_excel(writer, sheet_name="Formation_Spatial_Modifiers", index=False)
    fs.to_excel(writer, sheet_name="Spatial_Modifiers_Normalized", index=False)
    analog.to_excel(writer, sheet_name="WFAS_Formation_Analogs_v4", index=False)
    risk.to_excel(writer, sheet_name="Risk_Decomposition_v4", index=False)
    mc_df.to_excel(writer, sheet_name="Monte_Carlo_v4_LogOdds", index=False)
    pd.DataFrame([{"Parameter": k, "Weight": v} for k,v in WFAS_WEIGHTS.items()] + [{"Parameter":"Spatial_Support_0_1", "Weight":SPATIAL_WEIGHT}]).to_excel(writer, sheet_name="WFAS_Weights_v4", index=False)

# -----------------------------
# 5) Console summary
# -----------------------------
print("\n=== Top Formation Analogs — WFAS v4 ===")
cols = ["Target_Formation", "Analog_Basin", "Analog_Formation", "WFAS_v4", "Spatial_Support_0_1"]
print(analog[cols].head(12).to_string(index=False))

print("\n=== Risk Decomposition v4 ===")
print(risk[["Formation", "Seal_Risk_v4", "Timing_Risk_v4", "Leakage_Risk_v4", "Data_Gap_Risk_v4", "Fault_Risk_v4", "Overall_Risk_v4"]].to_string(index=False))

print("\n=== Monte Carlo v4 Log-Odds Results ===")
print(mc_df[["Formation", "Analog_support_WFAS", "P_source_mean", "P_reservoir_mean", "P_trap_mean", "Risk_mean", "P_petroleum_mean", "P_petroleum_p025", "P_petroleum_p975", "P_petroleum_gt_0_3", "P_petroleum_gt_0_5"]].to_string(index=False))

print(f"\nFiles written to ./{OUTPUT_DIR}/")
print(f"Workbook written: {OUTPUT_XLSX}")
