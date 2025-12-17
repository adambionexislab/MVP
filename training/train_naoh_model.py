import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# 1. Load data
# -----------------------------
DATA_PATH = "data/cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Features & target
# -----------------------------
FEATURES = ["pH(1)", "PAC"]
TARGET = "NAOH"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# 3. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. XGBoost (monotonic)
# -----------------------------
# pH ↑ → NAOH ↓
# PAC ↑ → NAOH ↑
monotone_constraints = "(-1,1)"

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    monotone_constraints=monotone_constraints,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print(f"NAOH model R²:  {r2_score(y_test, y_pred):.3f}")
print(f"NAOH model MAE: {mean_absolute_error(y_test, y_pred):.3f}")

# -----------------------------
# 6. Save model
# -----------------------------
joblib.dump(model, "models/naoh_model.pkl")
print("Saved models/naoh_model.pkl")
