import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/cleaned_data.csv")

# -----------------------------
# Features & target
# -----------------------------
FEATURES = [
    "Turbidity(1)",
    "TOC(1)",
    "COD(1)",
    "PAC",
    "FLOCCULANT"
]
TARGET = "Turbidity(2)"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
pred = model.predict(X_test)
print("Turbidity(2) R²:", r2_score(y_test, pred))
print("Turbidity(2) MAE:", mean_absolute_error(y_test, pred))

# -----------------------------
# Save
# -----------------------------
joblib.dump(model, "models/turb2_model.pkl")
print("Saved models/turb2_model.pkl")
