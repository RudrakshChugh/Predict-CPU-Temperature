import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from datetime import datetime

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import clone
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb

# Attempt to import CatBoost, skip if not found
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[!] CatBoost not found. Run 'pip install catboost' to include it.")

# ============================================================
# CONFIG
# ============================================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_BASE_DIR, "data", "combined_system_data.csv")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
OVERHEAT_THRESHOLD = 80.0
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# UTILS & PREPROCESSING
# ============================================================
def normalize_scores(x):
    """Normalize array to 0-1 range for AUC/Calibration calculation."""
    if x.max() == x.min(): return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

def expected_calibration_error(y, p, bins=10):
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.any():
            ece += abs(y[m].mean() - p[m].mean()) * m.mean()
    return ece

def create_features(df):
    """Temporal and Lag Feature Engineering."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Lag features (Crucial for Linear models to understand trends)
    for i in [1, 2, 5]:
        df[f'temp_lag_{i}'] = df['cpu_temp'].shift(i)
    
    # Moving Average
    df['temp_rolling_5'] = df['cpu_temp'].rolling(window=5).mean()
    
    # Fill NaNs created by lagging
    df = df.ffill().bfill()
    return df

# ============================================================
# DATA PREPARATION
# ============================================================
if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found.")
    sys.exit()

print("[*] Loading and Engineering Data...")
df_raw = pd.read_csv(CSV_PATH)
df = create_features(df_raw)

y = df["cpu_temp"]
X = df.drop(columns=["cpu_temp", "timestamp", "system_id"], errors="ignore")

# Temporal Train/Test Split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale features (ESSENTIAL for Linear/Ridge Regression)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# --- SAVE THE SCALER ---
# You MUST save the scaler so future input data can be scaled the exact same way!
scaler_path = os.path.join(RESULTS_DIR, "data_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[✔] Global Scaler saved to: {scaler_path}")

# ============================================================
# MODEL SELECTION
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
        tree_method="hist", n_estimators=500, learning_rate=0.03,
        max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=63,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
}

if HAS_CATBOOST:
    models["CatBoost"] = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6, 
        random_state=RANDOM_STATE, verbose=0, thread_count=-1
    )

# ============================================================
# RUN PIPELINE
# ============================================================
for name, model in models.items():
    print(f"\n[*] Training {name}...")
    out = os.path.join(RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)

    # 1. Fit
    model.fit(X_train_scaled, y_train)

    # --- SAVE THE MODEL (.pkl format) ---
    model_path = os.path.join(out, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"    [✔] Model weights saved to: {model_path}")

    # 2. Predict
    tr_pred = model.predict(X_train_scaled)
    te_pred = model.predict(X_test_scaled)

    # Regression Metrics
    r2_te = r2_score(y_test, te_pred)
    mae = mean_absolute_error(y_test, te_pred)
    rmse = np.sqrt(mean_squared_error(y_test, te_pred))

    # Confidence Score Calculation
    errors = np.abs(y_test - te_pred)
    within_2deg = (errors <= 2).mean() * 100
    conf_score = round(r2_te * 50 + max(0, (3 - mae) / 3 * 30) + within_2deg * 0.2, 1)

    # 3. Classification Performance (Threshold Based)
    y_true_cls = (y_test >= OVERHEAT_THRESHOLD).astype(int)
    y_pred_cls = (te_pred >= OVERHEAT_THRESHOLD).astype(int)
    scores_norm = normalize_scores(te_pred)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    try:
        roc = roc_auc_score(y_true_cls, scores_norm)
        pr_auc = average_precision_score(y_true_cls, scores_norm)
    except:
        roc, pr_auc = 0.5, 0.0

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-9)
    ece = expected_calibration_error(y_true_cls.values, scores_norm)

    # ------------------ SAVE METRICS ------------------
    with open(f"{out}/metrics.txt", "w") as f:
        f.write(f"MODEL PERFORMANCE REPORT: {name}\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"--- REGRESSION ---\n")
        f.write(f"R2 Score (Test):  {r2_te:.4f}\n")
        f.write(f"MAE:              {mae:.2f} °C\n")
        f.write(f"RMSE:             {rmse:.2f} °C\n")
        f.write(f"Confidence Score: {conf_score}%\n\n")
        f.write(f"--- CLASSIFICATION (Threshold: {OVERHEAT_THRESHOLD}°C) ---\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Specificity: {spec:.4f}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write(f"ROC-AUC:     {roc:.4f}\n")
        f.write(f"PR-AUC:      {pr_auc:.4f}\n")
        f.write(f"Calibration (ECE): {ece:.4f}\n")

    # ------------------ PLOTS (PDF) ------------------
    # 1. Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, te_pred, alpha=0.2, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"{name}: Regression Accuracy")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{out}/Actual_vs_Predicted.pdf"); plt.close()

    # 2. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", 
                xticklabels=['Normal', 'Overheat'], yticklabels=['Normal', 'Overheat'])
    plt.title(f"{name}: Overheat Detection CM")
    plt.savefig(f"{out}/Confusion.pdf"); plt.close()

    # 3. Feature Importance (Handles Coefs for Linear models)
    plt.figure(figsize=(8, 6))
    imp = None
    title_label = "Feature Importance"
    
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models use coefficients; we use absolute values to show impact
        imp = np.abs(model.coef_)
        title_label = "Feature Coefficients (Absolute)"
    elif name == "CatBoost":
        imp = model.get_feature_importance()

    if imp is not None:
        pd.Series(imp, index=X_train.columns).sort_values().tail(12).plot(kind='barh', color='teal')
        plt.title(f"{name}: {title_label}")
        plt.tight_layout()
        plt.savefig(f"{out}/Feature_Importance.pdf"); plt.close()

    # 4. Calibration
    plt.figure(figsize=(7, 6))
    if len(np.unique(y_true_cls)) > 1:
        fop, mpv = calibration_curve(y_true_cls, scores_norm, n_bins=10)
        plt.plot(mpv, fop, marker="s", color='black', label=name)
    plt.plot([0, 1], [0, 1], "--", color='gray')
    plt.title(f"{name}: Calibration Curve")
    plt.savefig(f"{out}/Calibration.pdf"); plt.close()

    # 5. Residuals Distribution Histogram
    plt.figure(figsize=(7, 6))
    sns.histplot(y_test - te_pred, kde=True, bins=50, color='purple')
    plt.title(f"{name}: Residuals Distribution")
    plt.xlabel("Residuals (Actual - Predicted) (°C)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{out}/Residuals_Distribution.pdf"); plt.close()

    # 6. Residuals vs Predicted (Homoscedasticity Check)
    plt.figure(figsize=(7, 6))
    plt.scatter(te_pred, y_test - te_pred, alpha=0.2, color='darkorange')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.title(f"{name}: Residuals vs Predicted")
    plt.xlabel("Predicted Temperature (°C)")
    plt.ylabel("Residuals (°C)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{out}/Residuals_vs_Predicted.pdf"); plt.close()

    # 7. Time Series Prediction Overlay (First 500 Test Samples for readability)
    plt.figure(figsize=(12, 5))
    subset_size = min(500, len(y_test))
    plt.plot(y_test.values[:subset_size], label="Actual Temp", color='royalblue', alpha=0.7)
    plt.plot(te_pred[:subset_size], label="Predicted Temp", color='crimson', alpha=0.7, linestyle='--')
    plt.axhline(OVERHEAT_THRESHOLD, color='black', linestyle=':', label='Overheat Threshold')
    plt.title(f"{name}: Time Series Overlay (First {subset_size} Test Samples)")
    plt.xlabel("Test Sample Index (Chronological)")
    plt.ylabel("CPU Temperature (°C)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out}/Time_Series_Overlay.pdf"); plt.close()

    # 8. ROC Curve
    plt.figure(figsize=(7, 6))
    if len(np.unique(y_true_cls)) > 1:
        fpr, tpr, _ = roc_curve(y_true_cls, scores_norm)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name}: Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{out}/ROC_Curve.pdf"); plt.close()

    # 9. Precision-Recall Curve
    plt.figure(figsize=(7, 6))
    if len(np.unique(y_true_cls)) > 1:
        pr, rc, _ = precision_recall_curve(y_true_cls, scores_norm)
        plt.plot(rc, pr, color='forestgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name}: Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{out}/PR_Curve.pdf"); plt.close()

    print(f"    [✔] {name} fully processed.")

print("\n" + "="*50)
print("[✔] FULL HYBRID PIPELINE COMPLETE")
print(f"[*] All Models and plots saved in: {RESULTS_DIR}")
print("="*50)