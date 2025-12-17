from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse

from optimization.ph_control import adjust_naoh, estimate_ph

# -----------------------------
# Load trained models
# -----------------------------
pac_model = joblib.load("models/pac_model.pkl")
floc_model = joblib.load("models/floc_model.pkl")
naoh_model = joblib.load("models/naoh_model.pkl")

# -----------------------------
# FastAPI instance
# -----------------------------
app = FastAPI(title="Water Treatment Dose Recommender")

# -----------------------------
# Input schema
# -----------------------------
class WaterInput(BaseModel):
    Turbidity: float
    TOC: float
    COD: float
    pH: float

# -----------------------------
# Helper function
# -----------------------------
def predict_dose(raw_water: dict):
    # PAC
    X_pac = pd.DataFrame([{
        "Turbidity(1)": raw_water["Turbidity"],
        "TOC(1)": raw_water["TOC"],
        "COD(1)": raw_water["COD"]
    }])
    pac = float(pac_model.predict(X_pac)[0])

    # FLOCCULANT
    X_floc = pd.DataFrame([{
        "Turbidity(1)": raw_water["Turbidity"],
        "TOC(1)": raw_water["TOC"],
        "COD(1)": raw_water["COD"],
        "PAC": pac
    }])
    floc = float(floc_model.predict(X_floc)[0])

    # NAOH
    X_naoh = pd.DataFrame([{
        "pH(1)": raw_water["pH"],
        "PAC": pac
    }])
    naoh_base = float(naoh_model.predict(X_naoh)[0])

    # pH estimate
    ph_est = estimate_ph(raw_water["pH"], pac, naoh_base)

    # NAOH safety adjustment
    naoh = adjust_naoh(
        naoh_base=naoh_base,
        ph1=raw_water["pH"],
        pac=pac,
        ph_pred=ph_est
    )

    return {
        "PAC": round(pac, 3),
        "FLOCCULANT": round(floc, 3),
        "NAOH": round(naoh, 3),
        "estimated_pH": round(ph_est, 3)
    }

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/predict-dose")
def predict(input_data: WaterInput):
    raw_dict = {
        "Turbidity": input_data.Turbidity,
        "TOC": input_data.TOC,
        "COD": input_data.COD,
        "pH": input_data.pH
    }

    result = predict_dose(raw_dict)

    warnings = []
    if result["estimated_pH"] < 7.0 or result["estimated_pH"] > 7.5:
        warnings.append("Estimated pH out of regulatory limits.")

    return {"dose": result, "warnings": warnings}

# -----------------------------
# UI endpoint (ROOT)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
