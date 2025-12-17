import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================
# Load models and scalers
# =============================
MODEL_DIR = "models"

model_ph = joblib.load(f"{MODEL_DIR}/model_ph.pkl")
scaler_ph = joblib.load(f"{MODEL_DIR}/scaler_ph.pkl")

model_turb = joblib.load(f"{MODEL_DIR}/model_turbidity.pkl")
model_toc = joblib.load(f"{MODEL_DIR}/model_toc.pkl")
model_cod = joblib.load(f"{MODEL_DIR}/model_cod.pkl")
scaler_solids = joblib.load(f"{MODEL_DIR}/scaler_solids.pkl")

# =============================
# Regulatory limits (HARD)
# =============================
LIMITS = {
    "pH_min": 7.0,
    "pH_max": 7.5,
    "TOC_max": 1.6,
    "COD_max": 140.0,
    "Turbidity_max": 6.0
}

# =============================
# Operator targets (from data)
# =============================
TARGETS = {
    "pH": 7.2,
    "TOC": 1.5,
    "COD": 120.0,
    "Turbidity": 5.0
}

# =============================
# Dose bounds
# =============================
BOUNDS = [
    (0.2, 20.0),   # PAC
    (0.1, 20.0),   # NAOH
    (2.0, 50.0)    # FLOCCULANT
]

# =============================
# NAOH dynamic parameters
# =============================
NAOH_PER_PAC = 1.15  # NaOH per unit PAC
PH_COEFF = 0.5   # NaOH per pH deviation from 7

# =============================
# Prediction function
# =============================
def predict_treated(x, raw_water):
    """
    Predict treated water quality with monotonic dose-response correction.
    
    Inputs:
        x: [PAC, NAOH, FLOCCULANT]
        raw_water: [Turbidity(1), pH(1), TOC(1), COD(1)]
    
    Returns:
        dict with predicted pH, TOC, COD, Turbidity
    """
    pac, naoh, floc = x
    turb1, ph1, toc1, cod1 = raw_water

    # ---- pH prediction (ML model) ----
    X_ph = pd.DataFrame([[ph1, pac, naoh]], columns=["pH(1)", "PAC", "NAOH"])
    X_ph_s = scaler_ph.transform(X_ph)
    ph2 = float(model_ph.predict(X_ph_s)[0])

    # ---- Solids prediction (ML model) ----
    X_sol = pd.DataFrame([[turb1, toc1, cod1, pac, floc]],
                         columns=["Turbidity(1)", "TOC(1)", "COD(1)", "PAC", "FLOCCULANT"])
    X_sol_s = scaler_solids.transform(X_sol)

    turb2 = float(model_turb.predict(X_sol_s)[0])
    toc2  = float(model_toc.predict(X_sol_s)[0])
    cod2  = float(model_cod.predict(X_sol_s)[0])

    # ---- Monotonic dose-response correction ----
    # Ensure more PAC/FLOCCULANT -> lower solids
    # Coefficients can be tuned to match process sensitivity
    turb2 = max(0.0, turb2 - 0.03 * (pac + floc))
    toc2  = max(0.0, toc2  - 0.03 * (pac + floc))
    cod2  = max(0.0, cod2  - 0.03 * (pac + floc))

    return {"pH": ph2, "TOC": toc2, "COD": cod2, "Turbidity": turb2}
# =============================
# Objective function
# =============================
def objective(x, raw_water):
    pred = predict_treated(x, raw_water)

    # Match operator-achieved quality
    turb_penalty = (pred["Turbidity"] - TARGETS["Turbidity"]) ** 2
    toc_penalty = (pred["TOC"] - TARGETS["TOC"]) ** 2
    cod_penalty = (pred["COD"] - TARGETS["COD"]) ** 2
    ph_penalty = (pred["pH"] - TARGETS["pH"]) ** 2

    # Minimize chemicals (small weight)
    dose_penalty = 0.05 * (x[0] + x[1] + x[2])

    return turb_penalty + toc_penalty + cod_penalty + ph_penalty + dose_penalty

# =============================
# Constraints
# =============================
def naoh_dynamic_constraint(x, raw_water):
    pac, naoh, _ = x
    ph1 = raw_water[1]

    ph_adjustment = PH_COEFF * (7 - ph1)
    pac_requirement = NAOH_PER_PAC * pac
    required_naoh = pac_requirement + ph_adjustment

    return naoh - required_naoh  # Can be negative if pH > 7

def constraint_funcs(raw_water):
    turb1, ph1, toc1, cod1 = raw_water

    def pH_min(x):
        return predict_treated(x, raw_water)["pH"] - LIMITS["pH_min"]

    def pH_max(x):
        return LIMITS["pH_max"] - predict_treated(x, raw_water)["pH"]

    def TOC_max(x):
        return LIMITS["TOC_max"] - predict_treated(x, raw_water)["TOC"]

    def COD_max(x):
        return LIMITS["COD_max"] - predict_treated(x, raw_water)["COD"]

    def Turb_max(x):
        return LIMITS["Turbidity_max"] - predict_treated(x, raw_water)["Turbidity"]

    def floc_ge_pac(x):
        pac, _, floc = x
        return floc - 7.0 * pac

    def pac_minimum(x):
        pac, _, _ = x
        load = 0.04 * turb1 + 0.04 * toc1 + 0.04 * cod1
        return pac - 0.0225 * load

    return [
        {"type": "ineq", "fun": pH_min},
        {"type": "ineq", "fun": pH_max},
        {"type": "ineq", "fun": TOC_max},
        {"type": "ineq", "fun": COD_max},
        {"type": "ineq", "fun": Turb_max},
        {"type": "ineq", "fun": lambda x: naoh_dynamic_constraint(x, raw_water)},
        {"type": "ineq", "fun": floc_ge_pac},
        {"type": "ineq", "fun": pac_minimum},
    ]

# =============================
# Main optimization function
# =============================
def optimize_dose(raw_water):
    initial_guesses = [
        [1.0, 0.5, 5.0],
        [5.0, 2.0, 10.0],
        [10.0, 5.0, 25.0]
    ]

    best = None
    for x0 in initial_guesses:
        res = minimize(
            objective,
            x0,
            args=(raw_water,),
            method="SLSQP",
            bounds=BOUNDS,
            constraints=constraint_funcs(raw_water),
            options={"maxiter": 300, "ftol": 1e-6}
        )
        if res.success:
            if best is None or res.fun < best.fun:
                best = res

    if best is None:
        return {"error": "No feasible solution found"}

    doses = best.x
    pred = predict_treated(doses, raw_water)

    return {
        "PAC": float(doses[0]),
        "NAOH": float(doses[1]),
        "FLOCCULANT": float(doses[2]),
        "predicted_output": pred
    }

# =============================
# Test run
# =============================
if __name__ == "__main__":
    example_raw = [70, 7.2, 10.57, 1079]  # Turbidity(1), pH(1), TOC(1), COD(1)
    result = optimize_dose(example_raw)
    print(result)
