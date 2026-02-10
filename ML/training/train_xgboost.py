# ============================================================
# CPU THERMAL MODEL – FULL FILE STYLE (FINAL)
# SAME OUTPUT AS Results-FT-Full-file
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import xgboost as xgb
import lightgbm as lgb

# Optional CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR = "/workspace/Akshat"
CSV_FILE = os.path.join(PROJECT_DIR, "combined_system_data.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results-FT-Full-file")

TARGET = "cpu_temp"
OVERHEAT_THRESHOLD = 80.0
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def expected_calibration_error(y, p, bins=10):
    ece = 0.0
    bins_edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > bins_edges[i]) & (p <= bins_edges[i+1])
        if m.any():
            ece += abs(y[m].mean() - p[m].mean()) * m.mean()
    return ece

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp_unix"] = df["timestamp"].astype("int64") // 10**9

FEATURES = [c for c in df.columns if c not in [TARGET, "timestamp"]]

X = df[FEATURES].ffill().bfill()
y = df[TARGET]

# Remove extreme glitches only
mask = y.diff().abs().fillna(0) <= 20
X, y = X[mask], y[mask]

# ============================================================
# SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
)

# ============================================================
# CATEGORICAL HANDLING
# ============================================================
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in FEATURES if c not in cat_cols]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_train_cat = pd.DataFrame(
    encoder.fit_transform(X_train[cat_cols]),
    columns=cat_cols, index=X_train.index
) if cat_cols else pd.DataFrame(index=X_train.index)

X_test_cat = pd.DataFrame(
    encoder.transform(X_test[cat_cols]),
    columns=cat_cols, index=X_test.index
) if cat_cols else pd.DataFrame(index=X_test.index)

scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train[num_cols]),
    columns=num_cols, index=X_train.index
)
X_test_num = pd.DataFrame(
    scaler.transform(X_test[num_cols]),
    columns=num_cols, index=X_test.index
)

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test  = pd.concat([X_test_num, X_test_cat], axis=1)

FEATURES_FINAL = X_train.columns.tolist()

# ============================================================
# MODELS
# ============================================================
models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.7,
        random_state=RANDOM_STATE, n_jobs=-1,
        tree_method="hist"
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=64,
        random_state=RANDOM_STATE, verbose=-1
    )
}

if HAS_CATBOOST:
    models["CatBoost"] = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6,
        random_state=RANDOM_STATE, verbose=0
    )

# ============================================================
# RUN PIPELINE
# ============================================================
for name, model in models.items():
    print(f"\n[*] Training {name}")
    out = os.path.join(RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)

    model.fit(X_train, y_train)

    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)

    # ---------------- REGRESSION ----------------
    r2 = r2_score(y_test, te_pred)
    mae = mean_absolute_error(y_test, te_pred)
    rmse = np.sqrt(mean_squared_error(y_test, te_pred))

    errors = np.abs(y_test - te_pred)
    within_2 = (errors <= 2).mean() * 100
    confidence = round(r2 * 50 + max(0, (3 - mae) / 3 * 30) + within_2 * 0.2, 1)

    # ---------------- CLASSIFICATION ----------------
    y_true = (y_test >= OVERHEAT_THRESHOLD).astype(int)
    y_pred = (te_pred >= OVERHEAT_THRESHOLD).astype(int)
    scores = normalize(te_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-9)
    ece = expected_calibration_error(y_true.values, scores)

    # ---------------- SAVE METRICS ----------------
    with open(f"{out}/metrics.txt", "w") as f:
        f.write(f"MODEL PERFORMANCE REPORT: {name}\n")
        f.write("="*50 + "\n\n")
        f.write("--- REGRESSION ---\n")
        f.write(f"R2 Score (Test):  {r2:.4f}\n")
        f.write(f"MAE:              {mae:.2f} °C\n")
        f.write(f"RMSE:             {rmse:.2f} °C\n")
        f.write(f"Confidence Score: {confidence}%\n\n")
        f.write(f"--- CLASSIFICATION (Threshold: {OVERHEAT_THRESHOLD}°C) ---\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Specificity: {spec:.4f}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write(f"ROC-AUC:     {roc:.4f}\n")
        f.write(f"PR-AUC:      {pr:.4f}\n")
        f.write(f"Calibration (ECE): {ece:.4f}\n")

    # ---------------- PLOTS ----------------
    # Actual vs Predicted
    plt.figure(figsize=(7,6))
    plt.scatter(y_test, te_pred, alpha=0.25)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.xlabel("Actual (°C)")
    plt.ylabel("Predicted (°C)")
    plt.title(f"{name}: Regression Accuracy")
    plt.savefig(f"{out}/Actual_vs_Predicted.pdf")
    plt.close()

    # Confusion
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Normal","Overheat"],
                yticklabels=["Normal","Overheat"])
    plt.title(f"{name}: Confusion Matrix")
    plt.savefig(f"{out}/Confusion.pdf")
    plt.close()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    elif name == "CatBoost":
        imp = model.get_feature_importance()
    else:
        imp = None

    if imp is not None:
        pd.Series(imp, index=FEATURES_FINAL).sort_values().plot.barh(
            figsize=(8,6)
        )
        plt.title(f"{name}: Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{out}/Feature_Importance.pdf")
        plt.close()

    # Calibration
    plt.figure(figsize=(7,6))
    fop, mpv = calibration_curve(y_true, scores, n_bins=10)
    plt.plot(mpv, fop, marker="s")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title(f"{name}: Calibration Curve")
    plt.savefig(f"{out}/Calibration.pdf")
    plt.close()

    print(f"[✔] {name} complete")

print("\n[✔] FULL PIPELINE COMPLETE – OUTPUT MATCHES Results-FT-Full-file")
