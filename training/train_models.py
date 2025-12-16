# training/train_models.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/cleaned_data.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

EXPECTED_COLUMNS = [
    "Turbidity(1)", "pH(1)", "TOC(1)", "COD(1)",
    "PAC", "NAOH", "FLOCCULANT",
    "pH(2)", "TOC(2)", "COD(2)", "Turbidity(2)"
]

missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# -----------------------------
# Feature / Target split
# -----------------------------
X = df[
    ["Turbidity(1)", "pH(1)", "TOC(1)", "COD(1)", "PAC", "NAOH", "FLOCCULANT"]
].values

y_pH = df["pH(2)"].values
y_TOC = df["TOC(2)"].values
y_COD = df["COD(2)"].values
y_Turb = df["Turbidity(2)"].values

# -----------------------------
# Train / test split
# -----------------------------
(
    X_train,
    X_test,
    y_pH_train,
    y_pH_test,
    y_TOC_train,
    y_TOC_test,
    y_COD_train,
    y_COD_test,
    y_Turb_train,
    y_Turb_test,
) = train_test_split(
    X,
    y_pH,
    y_TOC,
    y_COD,
    y_Turb,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# -----------------------------
# Model factory
# -----------------------------
def create_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# -----------------------------
# Train models
# -----------------------------
models = {
    "pH": create_model(),
    "TOC": create_model(),
    "COD": create_model(),
    "Turbidity": create_model(),
}

models["pH"].fit(X_train_scaled, y_pH_train)
models["TOC"].fit(X_train_scaled, y_TOC_train)
models["COD"].fit(X_train_scaled, y_COD_train)
models["Turbidity"].fit(X_train_scaled, y_Turb_train)

# -----------------------------
# Evaluation
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\nValidation RMSE:")
print("-" * 30)

for name, model in models.items():
    if name == "pH":
        y_test = y_pH_test
    elif name == "TOC":
        y_test = y_TOC_test
    elif name == "COD":
        y_test = y_COD_test
    else:
        y_test = y_Turb_test

    preds = model.predict(X_test_scaled)
    score = rmse(y_test, preds)

    print(f"{name}: {score:.3f}")

# -----------------------------
# Save models
# -----------------------------
joblib.dump(models["pH"], f"{MODEL_DIR}/model_pH.pkl")
joblib.dump(models["TOC"], f"{MODEL_DIR}/model_TOC.pkl")
joblib.dump(models["COD"], f"{MODEL_DIR}/model_COD.pkl")
joblib.dump(models["Turbidity"], f"{MODEL_DIR}/model_Turbidity.pkl")

print("\nModels and scaler saved successfully.")
