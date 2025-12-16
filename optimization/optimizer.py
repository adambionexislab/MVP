import joblib
import numpy as np
from scipy.optimize import minimize

# -----------------------------
# Load models and scaler
# -----------------------------
MODEL_DIR = "models"

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
model_pH = joblib.load(f"{MODEL_DIR}/model_pH.pkl")
model_TOC = joblib.load(f"{MODEL_DIR}/model_TOC.pkl")
model_COD = joblib.load(f"{MODEL_DIR}/model_COD.pkl")
model_Turb = joblib.load(f"{MODEL_DIR}/model_Turbidity.pkl")

# -----------------------------
# Regulatory limits (HARD)
# -----------------------------
LIMITS = {
    "pH_min": 7.0,
    "pH_max": 7.5,
    "TOC_max": 10.0,
    "COD_max": 200.0,
    "Turbidity_max": 10.0
}

# -----------------------------
# Dose bounds (PROCESS-BASED)
# -----------------------------
BOUNDS = [
    (0.2, 20),   # PAC
    (0.1, 20),   # NAOH
    (2, 50)      # FLOCCULANT (diluted)
]

# PAC–NAOH coupling coefficient
NAOH_PER_PAC = 1.15

# Optional: historical dose prior
HISTORICAL_MEAN = np.array([10, 2, 10])  # PAC, NAOH, FLOCCULANT
DOSE_PRIOR_WEIGHT = 0.01

# Safety margin for target
SAFE_MARGIN = 0.85  # 99% of limit
# -----------------------------
# Prediction helper
# -----------------------------
def predict_treated(x, raw_water):
    features = np.array([
        raw_water[0],  # Turbidity(1)
        raw_water[1],  # pH(1)
        raw_water[2],  # TOC(1)
        raw_water[3],  # COD(1)
        x[0],          # PAC
        x[1],          # NAOH
        x[2]           # FLOCCULANT
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)

    return {
        "pH": float(model_pH.predict(features_scaled)[0]),
        "TOC": float(model_TOC.predict(features_scaled)[0]),
        "COD": float(model_COD.predict(features_scaled)[0]),
        "Turbidity": float(model_Turb.predict(features_scaled)[0])
    }

# -----------------------------
# Margin-based objective
# -----------------------------
def objective(x, raw_water):
    pred = predict_treated(x, raw_water)

    # Soft margin targets (below regulatory limits)
    turb_target = SAFE_MARGIN * LIMITS["Turbidity_max"]
    toc_target = SAFE_MARGIN * LIMITS["TOC_max"]
    cod_target = SAFE_MARGIN * LIMITS["COD_max"]
    ph_target = 7.25  # middle of pH band

    # Penalize distance from soft targets
    turb_penalty = (pred["Turbidity"] - turb_target) ** 2
    toc_penalty = (pred["TOC"] - toc_target) ** 2
    cod_penalty = (pred["COD"] - cod_target) ** 2
    ph_penalty = (pred["pH"] - ph_target) ** 2

    # Encourage staying near historical dose if desired
    dose_penalty = DOSE_PRIOR_WEIGHT * np.sum((x - HISTORICAL_MEAN) ** 2)

    return turb_penalty + toc_penalty + cod_penalty + ph_penalty + dose_penalty

# -----------------------------
# Constraints
# -----------------------------
def constraint_funcs(raw_water):

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

    def naoh_pac_coupling(x):
        pac, naoh, _ = x
        return naoh - NAOH_PER_PAC * pac

    return [
        {"type": "ineq", "fun": pH_min},
        {"type": "ineq", "fun": pH_max},
        {"type": "ineq", "fun": TOC_max},
        {"type": "ineq", "fun": COD_max},
        {"type": "ineq", "fun": Turb_max},
        {"type": "ineq", "fun": naoh_pac_coupling}
    ]

# -----------------------------
# Main optimization function
# -----------------------------
def optimize_dose(raw_water):

    initial_guesses = [
        [0.5, 0.2, 3],
        [5, 2, 10],
        [10, 5, 25]
    ]

    best_result = None

    for x0 in initial_guesses:
        result = minimize(
            objective,
            x0,
            args=(raw_water,),
            method="SLSQP",
            bounds=BOUNDS,
            constraints=constraint_funcs(raw_water),
            options={"maxiter": 300, "ftol": 1e-6}
        )

        if result.success:
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        return {"error": "No feasible dosage found within constraints"}

    doses = best_result.x
    treated = predict_treated(doses, raw_water)

    return {
        "PAC": float(doses[0]),
        "NAOH": float(doses[1]),
        "FLOCCULANT": float(doses[2]),
        "predicted_output": treated
    }

# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    example_raw = [250, 6.2, 1531.8, 2280]
    result = optimize_dose(example_raw)
    print(result)
