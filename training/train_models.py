import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/cleaned_data.csv")

# -----------------------------
# A. pH model (NaOH neutralization)
# -----------------------------
X_ph = df[
    ["pH(1)", "PAC", "NAOH"]
]
y_ph = df["pH(2)"]

Xph_train, Xph_test, yph_train, yph_test = train_test_split(
    X_ph, y_ph, test_size=0.2, random_state=42
)

scaler_ph = StandardScaler()
Xph_train_s = scaler_ph.fit_transform(Xph_train)
Xph_test_s = scaler_ph.transform(Xph_test)

model_ph = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_ph.fit(Xph_train_s, yph_train)

# -----------------------------
# B. Solids models (PAC + FLOCCULANT)
# -----------------------------
X_solids = df[
    ["Turbidity(1)", "TOC(1)", "COD(1)", "PAC", "FLOCCULANT"]
]

targets = {
    "Turbidity": "Turbidity(2)",
    "TOC": "TOC(2)",
    "COD": "COD(2)"
}

Xsol_train, Xsol_test = train_test_split(
    X_solids, test_size=0.2, random_state=42
)

scaler_solids = StandardScaler()
Xsol_train_s = scaler_solids.fit_transform(Xsol_train)
Xsol_test_s = scaler_solids.transform(Xsol_test)

models_solids = {}

for name, target in targets.items():
    y = df.loc[Xsol_train.index, target]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(Xsol_train_s, y)
    models_solids[name] = model

# -----------------------------
# Save everything
# -----------------------------
joblib.dump(model_ph, "models/model_ph.pkl")
joblib.dump(scaler_ph, "models/scaler_ph.pkl")

joblib.dump(models_solids["Turbidity"], "models/model_turbidity.pkl")
joblib.dump(models_solids["TOC"], "models/model_toc.pkl")
joblib.dump(models_solids["COD"], "models/model_cod.pkl")
joblib.dump(scaler_solids, "models/scaler_solids.pkl")

print("✅ Models trained and saved successfully")
