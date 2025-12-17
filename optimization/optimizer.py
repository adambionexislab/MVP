import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import joblib  # for loading models
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load ML models and scalers
# -----------------------------
model_ph       = joblib.load("models/model_ph.pkl")
scaler_ph      = joblib.load("models/scaler_ph.pkl")
model_turb     = joblib.load("models/model_turbidity.pkl")
model_toc      = joblib.load("models/model_toc.pkl")
model_cod      = joblib.load("models/model_cod.pkl")
scaler_solids  = joblib.load("models/scaler_solids.pkl")

# -----------------------------
# 2. Define process limits
# -----------------------------
LIMITS = {
    "pH_min": 7.0,
    "pH_max": 7.5,
    "TOC_max": 10.0,
    "COD_max": 200.0,
    "Turbidity_max": 10.0
}

# -----------------------------
# 3. Predict treated water
# -----------------------------
def predict_treated(x, raw_water):
    pac, naoh, floc = x
    turb1, ph1, toc1, cod1 = raw_water

    # --- pH prediction ---
    X_ph = pd.DataFrame([[ph1, pac, naoh]], columns=["pH(1)", "PAC", "NAOH"])
    X_ph_s = scaler_ph.transform(X_ph)
    ph2 = float(model_ph.predict(X_ph_s)[0])

    # --- Solids prediction ---
    X_sol = pd.DataFrame([[turb1, toc1, cod1, pac, floc]],
                         columns=["Turbidity(1)", "TOC(1)", "COD(1)", "PAC", "FLOCCULANT"])
    X_sol_s = scaler_solids.transform(X_sol)

    turb2 = float(model_turb.predict(X_sol_s)[0])
    toc2  = float(model_toc.predict(X_sol_s)[0])
    cod2  = float(model_cod.predict(X_sol_s)[0])

    # --- Monotonic dose-response correction ---
    turb2 = max(0.0, turb2 - 0.05 * (pac + floc))
    toc2  = max(0.0, toc2  - 0.03 * (pac + floc))
    cod2  = max(0.0, cod2  - 0.04 * (pac + floc))

    return {"pH": ph2, "TOC": toc2, "COD": cod2, "Turbidity": turb2}

# -----------------------------
# 4. NAOH dynamic constraint
# -----------------------------
NAOH_PER_PAC = 1.15
def naoh_dynamic_constraint(x, raw_water):
    pac, naoh, _ = x
    turb1, ph1, toc1, cod1 = raw_water
    # 0.5 NaOH per 1 pH unit below/above 7 plus coupling with PAC
    naoh_required = NAOH_PER_PAC * (7 - ph1) + NAOH_PER_PAC * pac
    return naoh - naoh_required

# -----------------------------
# 5. PAC minimum & target
# -----------------------------
# Learned or estimated from historical data
PAC_COEFF = {"Turb": 0.005, "TOC": 0.0, "COD": 0.0005, "Intercept": 0.0}
PAC_WEIGHT = 5.0  # penalty for deviating from target

def pac_minimum(x, raw_water):
    pac, _, _ = x
    turb1, _, toc1, cod1 = raw_water
    pac_target = PAC_COEFF["Turb"] * turb1 + PAC_COEFF["TOC"] * toc1 + PAC_COEFF["COD"] * cod1 + PAC_COEFF["Intercept"]
    return pac - pac_target  # hard safety floor if needed

# -----------------------------
# 6. Other constraints
# -----------------------------
def floc_ge_pac(x):
    pac, _, floc = x
    return floc - pac  # simple constraint: floc >= PAC

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

    return [
        {"type": "ineq", "fun": pH_min},
        {"type": "ineq", "fun": pH_max},
        {"type": "ineq", "fun": TOC_max},
        {"type": "ineq", "fun": COD_max},
        {"type": "ineq", "fun": Turb_max},
        {"type": "ineq", "fun": lambda x: naoh_dynamic_constraint(x, raw_water)},
        {"type": "ineq", "fun": floc_ge_pac},
        {"type": "ineq", "fun": lambda x: pac_minimum(x, raw_water)}
    ]

# -----------------------------
# 7. Objective function
# -----------------------------
DOSE_WEIGHTS = {"PAC": 0.002, "NAOH": 0.02, "FLOCCULANT": 0.05}

def objective(x, raw_water):
    pac, naoh, floc = x
    pred = predict_treated(x, raw_water)

    # --- Dose penalty ---
    dose_penalty = DOSE_WEIGHTS["PAC"] * pac + DOSE_WEIGHTS["NAOH"] * naoh + DOSE_WEIGHTS["FLOCCULANT"] * floc

    # --- Quality penalty ---
    q_penalty = 0.0
    if pred["pH"] < LIMITS["pH_min"]:
        q_penalty += (LIMITS["pH_min"] - pred["pH"]) ** 2
    if pred["pH"] > LIMITS["pH_max"]:
        q_penalty += (pred["pH"] - LIMITS["pH_max"]) ** 2
    if pred["TOC"] > LIMITS["TOC_max"]:
        q_penalty += (pred["TOC"] - LIMITS["TOC_max"]) ** 2
    if pred["COD"] > LIMITS["COD_max"]:
        q_penalty += (pred["COD"] - LIMITS["COD_max"]) ** 2
    if pred["Turbidity"] > LIMITS["Turbidity_max"]:
        q_penalty += (pred["Turbidity"] - LIMITS["Turbidity_max"]) ** 2

    # --- PAC deviation penalty ---
    turb1, _, toc1, cod1 = raw_water
    pac_target = PAC_COEFF["Turb"] * turb1 + PAC_COEFF["TOC"] * toc1 + PAC_COEFF["COD"] * cod1 + PAC_COEFF["Intercept"]
    pac_penalty = PAC_WEIGHT * (pac - pac_target) ** 2

    return dose_penalty + q_penalty + pac_penalty

# -----------------------------
# 8. Optimizer
# -----------------------------
def optimize_dose(raw_water):
    # initial guess: 1/2 of typical dose ranges
    x0 = np.array([1.0, 1.0, 5.0])
    # bounds for PAC, NAOH, FLOCCULANT
    bounds = [(0.2, 20), (0.1, 20), (2, 50)]

    result = minimize(
        objective,
        x0,
        args=(raw_water,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraint_funcs(raw_water),
        options={"disp": False, "maxiter": 500}
    )

    pac_opt, naoh_opt, floc_opt = result.x
    treated_pred = predict_treated(result.x, raw_water)
    return {"PAC": pac_opt, "NAOH": naoh_opt, "FLOCCULANT": floc_opt, "predicted_output": treated_pred}

# -----------------------------
# 9. Test run
# -----------------------------
if __name__ == "__main__":
    # Example raw water input: [Turbidity, pH, TOC, COD]
    raw = [100, 7.1, 4.0, 120.0]
    result = optimize_dose(raw)
    print(result)

# -----------------------------
# 1. Test PAC linearity with Turbidity sweep
# -----------------------------
def test_pac_linearity():
    turbidity_values = np.linspace(50, 250, 10)
    pac_values = []

    for turb in turbidity_values:
        raw = [turb, 7.1, 4.0, 120.0]  # keep other inputs constant
        result = optimize_dose(raw)
        pac_values.append(result["PAC"])

    plt.figure()
    plt.plot(turbidity_values, pac_values, marker='o')
    plt.xlabel("Influent Turbidity")
    plt.ylabel("Optimized PAC")
    plt.title("PAC vs Turbidity (should be monotonic)")
    plt.grid(True)
    plt.show()

    print("PAC values:", pac_values)

# -----------------------------
# 2. Test pH response to NAOH sweep
# -----------------------------
def test_ph_linearity():
    naoh_values = np.linspace(0.1, 10.0, 10)
    ph_values = []

    for naoh in naoh_values:
        # example raw water and PAC/floc fixed
        raw = [100, 6.5, 4.0, 120.0]
        x = [3.0, naoh, 10.0]
        treated = optimize_dose(raw)["predicted_output"]
        ph_values.append(treated["pH"])

    plt.figure()
    plt.plot(naoh_values, ph_values, marker='o', color='r')
    plt.xlabel("NAOH Dose")
    plt.ylabel("Predicted pH")
    plt.title("pH vs NAOH (should increase roughly linearly)")
    plt.grid(True)
    plt.show()

    print("pH values:", ph_values)

# -----------------------------
# 3. Run tests
# -----------------------------
if __name__ == "__main__":
    test_pac_linearity()
    test_ph_linearity()