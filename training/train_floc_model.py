import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
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
FEATURES = ["Turbidity(1)", "TOC(1)", "COD(1)", "PAC"]
TARGET = "FLOCCULANT"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# 3. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. XGBoost with monotonic constraints
# -----------------------------
# Order matches FEATURES
# All inputs increase → FLOCCULANT increases
monotone_constraints = "(1,1,1,1)"

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
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

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"FLOCCULANT model R²:  {r2:.3f}")
print(f"FLOCCULANT model MAE: {mae:.3f}")

# -----------------------------
# 6. Save model
# -----------------------------
joblib.dump(model, "models/floc_model.pkl")
print("Saved models/floc_model.pkl")

# -----------------------------
# 7. Feature importance
# -----------------------------
importance = model.feature_importances_
for f, imp in zip(FEATURES, importance):
    print(f"{f}: {imp:.3f}")

# -----------------------------
# 8. Sanity plot: FLOCCULANT vs PAC
# -----------------------------
def plot_floc_vs_pac():
    pac_range = np.linspace(
        df["PAC"].quantile(0.05),
        df["PAC"].quantile(0.95),
        50
    )

    turb_med = df["Turbidity(1)"].median()
    toc_med = df["TOC(1)"].median()
    cod_med = df["COD(1)"].median()

    X_plot = pd.DataFrame({
        "Turbidity(1)": turb_med,
        "TOC(1)": toc_med,
        "COD(1)": cod_med,
        "PAC": pac_range
    })

    floc_pred = model.predict(X_plot)

    plt.figure()
    plt.plot(pac_range, floc_pred)
    plt.xlabel("PAC dose")
    plt.ylabel("Predicted FLOCCULANT dose")
    plt.title("FLOCCULANT vs PAC (monotonic check)")
    plt.grid(True)
    plt.show()

plot_floc_vs_pac()
