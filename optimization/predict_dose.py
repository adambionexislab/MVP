import joblib
import pandas as pd

from ph_control import adjust_naoh, estimate_ph

# -----------------------------
# Load models
# -----------------------------
pac_model = joblib.load("models/pac_model.pkl")
floc_model = joblib.load("models/floc_model.pkl")
naoh_model = joblib.load("models/naoh_model.pkl")

# -----------------------------
# Main prediction function
# -----------------------------
def predict_dose(raw_water: dict):
    """
    raw_water keys:
    Turbidity(1), TOC(1), COD(1), pH(1)
    """

    # -----------------------------
    # PAC
    # -----------------------------
    X_pac = pd.DataFrame([{
        "Turbidity(1)": raw_water["Turbidity(1)"],
        "TOC(1)": raw_water["TOC(1)"],
        "COD(1)": raw_water["COD(1)"]
    }])
    pac = float(pac_model.predict(X_pac)[0])

    # -----------------------------
    # FLOCCULANT
    # -----------------------------
    X_floc = pd.DataFrame([{
        "Turbidity(1)": raw_water["Turbidity(1)"],
        "TOC(1)": raw_water["TOC(1)"],
        "COD(1)": raw_water["COD(1)"],
        "PAC": pac
    }])
    floc = float(floc_model.predict(X_floc)[0])

    # -----------------------------
    # NAOH (policy)
    # -----------------------------
    X_naoh = pd.DataFrame([{
        "pH(1)": raw_water["pH(1)"],
        "PAC": pac
    }])
    naoh_base = float(naoh_model.predict(X_naoh)[0])

    # -----------------------------
    # pH control
    # -----------------------------
    X_naoh = pd.DataFrame([{
        "pH(1)": raw_water["pH"],
        "PAC": pac
    }])
    naoh = float(naoh_model.predict(X_naoh)[0])
    return {
        "PAC": round(pac, 3),
        "FLOCCULANT": round(floc, 3),
        "NAOH": round(naoh, 3),
        "estimated_pH": round(ph_est, 3)
    }

# -----------------------------
# test
# -----------------------------
if __name__ == "__main__":
    sample = {
        "Turbidity(1)": 250,
        "TOC(1)": 31.8,
        "COD(1)": 2280,
        "pH(1)": 7.2 }
    print(predict_dose(sample))
    
