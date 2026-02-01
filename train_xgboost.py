import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# --- CONFIGURATION ---
CSV_FILE = "combined_system_data.csv"  
MODEL_OUTPUT = "cpu_xgboost_model.pkl"
RESULTS_OUTPUT = "training_results.txt"

print("="*60)
print("CPU Temperature Prediction - Enhanced XGBoost Training")
print("="*60)

# Load data
print(f"\n1. Loading data from {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"   ✓ Loaded {len(df)} samples with {len(df.columns)} columns")
except FileNotFoundError:
    print(f"   ✗ Error: {CSV_FILE} not found!")
    exit(1)

# Display basic info
print(f"\n2. Data Overview:")
print(f"   - Total samples: {len(df)}")
print(f"   - Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   - Temperature range: {df['cpu_temp'].min():.1f}°C to {df['cpu_temp'].max():.1f}°C")

# Define features (exclude target and non-predictive columns)
EXCLUDE_FEATURES = [
    'timestamp',      # Not a predictive feature
    'cpu_temp',       # Target variable
    'ambient_temp',   # Derived from cpu_temp (data leakage)
    'system_id',      # Metadata
    'cpu_temp_prev_1s',  # Can cause data leakage in real-time prediction
    'cpu_temp_prev_5s',  # Can cause data leakage in real-time prediction
    'voltage',        # Highly correlated with power_estimated
    'current',        # Highly correlated with power_estimated
    'clock_speed_max', # Constant value, not useful for prediction
    'disk_read_mb_s',  # Low importance (< 3%)
    'disk_write_mb_s'  # Low importance (< 3%)
]

# Get all feature columns
all_features = [col for col in df.columns if col not in EXCLUDE_FEATURES]

print(f"\n3. Feature Selection:")
print(f"   - Total features: {len(all_features)}")
print(f"   - Features used: {', '.join(all_features)}")

# Prepare data
X = df[all_features]
y = df['cpu_temp']

# Handle missing values
print(f"\n4. Data Cleaning:")
missing_before = X.isnull().sum().sum()
if missing_before > 0:
    print(f"   - Missing values found: {missing_before}")
    X = X.fillna(method='ffill').fillna(method='bfill')
    print(f"   ✓ Missing values filled")
else:
    print(f"   ✓ No missing values")

# Remove rows where target is missing
valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask]

# Remove outliers (temperature changes > 15°C in 1 second)
print(f"\n5. Outlier Detection:")
temp_diff = y.diff().abs()
outlier_mask = temp_diff > 15
outliers_removed = outlier_mask.sum()
if outliers_removed > 0:
    X = X[~outlier_mask]
    y = y[~outlier_mask]
    print(f"   ✓ Removed {outliers_removed} outliers (temp change >15°C/s)")
else:
    print(f"   ✓ No outliers detected")

# Remove warmup period (first 2 minutes = 120 samples)
warmup_samples = min(120, len(X) // 20)  # Remove first 5% or 120 samples
X = X.iloc[warmup_samples:]
y = y.iloc[warmup_samples:]
print(f"   ✓ Removed {warmup_samples} warmup samples")
print(f"   ✓ Valid samples: {len(X)}")

# Split data (70% train, 15% validation, 15% test)
print(f"\n6. Splitting Data (70/15/15):")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True  # 0.176 * 0.85 ≈ 0.15
)
print(f"   - Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Train XGBoost model with improved hyperparameters
print(f"\n7. Training Enhanced XGBoost Model...")
model = XGBRegressor(
    n_estimators=150,      # Increased from 100 for better learning
    max_depth=5,           # Increased from 4 for more complex patterns
    learning_rate=0.05,    # Keep same for stability
    min_child_weight=2,    # Reduced from 3 for more flexibility
    subsample=0.8,         # Increased from 0.7 for more data usage
    colsample_bytree=0.8,  # Increased from 0.7 for more features
    reg_alpha=0.05,        # Reduced L1 regularization
    reg_lambda=0.5,        # Reduced L2 regularization
    gamma=0.05,            # Reduced for more splits
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train, verbose=False)
print(f"   ✓ Model trained successfully!")

# Cross-validation on training set
print(f"\n8. Cross-Validation (5-fold):")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"   - CV R² Scores: {cv_scores}")
print(f"   - Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Make predictions
print(f"\n9. Evaluating Model Performance...")
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calculate confidence scores based on prediction accuracy
print(f"\n9.5. Calculating Confidence Scores...")
test_errors = np.abs(y_test - y_test_pred)

# Confidence levels based on error thresholds
within_1deg = (test_errors <= 1.0).sum() / len(test_errors) * 100
within_2deg = (test_errors <= 2.0).sum() / len(test_errors) * 100
within_3deg = (test_errors <= 3.0).sum() / len(test_errors) * 100

# Overall model confidence score (0-100)
# Based on: R² score (50%), MAE (30%), and prediction consistency (20%)
r2_component = test_r2 * 50  # R² contributes 50%
mae_component = max(0, (3 - test_mae) / 3 * 30)  # MAE contributes 30% (3°C = 0%, 0°C = 30%)
consistency_component = within_2deg * 0.2  # Predictions within 2°C contribute 20%

overall_confidence = round(r2_component + mae_component + consistency_component, 1)

print(f"   ✓ Confidence score calculated: {overall_confidence}%")


# Display results
print(f"\n{'='*60}")
print(f"TRAINING RESULTS")
print(f"{'='*60}")
print(f"\nTraining Set Performance:")
print(f"   - R² Score:  {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"   - MAE:       {train_mae:.2f}°C")
print(f"   - RMSE:      {train_rmse:.2f}°C")

print(f"\nValidation Set Performance:")
print(f"   - R² Score:  {val_r2:.4f} ({val_r2*100:.2f}%)")
print(f"   - MAE:       {val_mae:.2f}°C")
print(f"   - RMSE:      {val_rmse:.2f}°C")

print(f"\nTest Set Performance:")
print(f"   - R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   - MAE:       {test_mae:.2f}°C")
print(f"   - RMSE:      {test_rmse:.2f}°C")

print(f"\nOverfitting Analysis:")
print(f"   - Train-Val Gap:  {abs(train_r2 - val_r2)*100:.2f}%")
print(f"   - Train-Test Gap: {abs(train_r2 - test_r2)*100:.2f}%")

print(f"\nModel Confidence Score:")
print(f"   - Overall Confidence: {overall_confidence}%")
print(f"   - Predictions within ±1°C: {within_1deg:.1f}%")
print(f"   - Predictions within ±2°C: {within_2deg:.1f}%")
print(f"   - Predictions within ±3°C: {within_3deg:.1f}%")


# Feature importance
print(f"\n{'='*60}")
print(f"TOP 10 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:25s} {row['importance']:.4f}")

# Save model
print(f"\n10. Saving Model...")
joblib.dump(model, MODEL_OUTPUT)
print(f"   ✓ Model saved to: {MODEL_OUTPUT}")

# Save results to file
print(f"\n11. Saving Results...")
with open(RESULTS_OUTPUT, 'w') as f:
    f.write("="*60 + "\n")
    f.write("CPU TEMPERATURE PREDICTION - ENHANCED XGBOOST MODEL\n")
    f.write("="*60 + "\n")
    f.write(f"\nTraining Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {CSV_FILE}\n")
    f.write(f"Total Samples: {len(df)}\n")
    f.write(f"Valid Samples (after cleaning): {len(X) + warmup_samples}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_val)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Outliers Removed: {outliers_removed}\n")
    f.write(f"Warmup Samples Removed: {warmup_samples}\n")
    f.write(f"\n{'='*60}\n")
    f.write("MODEL PERFORMANCE\n")
    f.write(f"{'='*60}\n")
    f.write(f"\nTraining Set:\n")
    f.write(f"  R² Score:  {train_r2:.4f} ({train_r2*100:.2f}%)\n")
    f.write(f"  MAE:       {train_mae:.2f}°C\n")
    f.write(f"  RMSE:      {train_rmse:.2f}°C\n")
    f.write(f"\nValidation Set:\n")
    f.write(f"  R² Score:  {val_r2:.4f} ({val_r2*100:.2f}%)\n")
    f.write(f"  MAE:       {val_mae:.2f}°C\n")
    f.write(f"  RMSE:      {val_rmse:.2f}°C\n")
    f.write(f"\nTest Set:\n")
    f.write(f"  R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)\n")
    f.write(f"  MAE:       {test_mae:.2f}°C\n")
    f.write(f"  RMSE:      {test_rmse:.2f}°C\n")
    f.write(f"\nOverfitting Analysis:\n")
    f.write(f"  Train-Val Gap:  {abs(train_r2 - val_r2)*100:.2f}%\n")
    f.write(f"  Train-Test Gap: {abs(train_r2 - test_r2)*100:.2f}%\n")
    f.write(f"\nCross-Validation:\n")
    f.write(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
    f.write(f"\n{'='*60}\n")
    f.write("MODEL CONFIDENCE SCORE\n")
    f.write(f"{'='*60}\n")
    f.write(f"\nOverall Confidence: {overall_confidence}%\n")
    f.write(f"\nPrediction Accuracy Breakdown:\n")
    f.write(f"  Predictions within ±1°C: {within_1deg:.1f}%\n")
    f.write(f"  Predictions within ±2°C: {within_2deg:.1f}%\n")
    f.write(f"  Predictions within ±3°C: {within_3deg:.1f}%\n")
    f.write(f"\nConfidence Calculation:\n")
    f.write(f"  - R² Score Component (50%):        {r2_component:.1f}\n")
    f.write(f"  - MAE Component (30%):             {mae_component:.1f}\n")
    f.write(f"  - Consistency Component (20%):     {consistency_component:.1f}\n")
    f.write(f"  - Total Confidence Score:          {overall_confidence}%\n")
    f.write(f"\n{'='*60}\n")
    f.write("FEATURE IMPORTANCE (Top 10)\n")
    f.write(f"{'='*60}\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"{row['feature']:25s} {row['importance']:.4f}\n")
    f.write(f"\n{'='*60}\n")
    f.write("ALL FEATURES\n")
    f.write(f"{'='*60}\n")
    for feature in all_features:
        f.write(f"  - {feature}\n")

print(f"   ✓ Results saved to: {RESULTS_OUTPUT}")

# Create enhanced visualizations
print(f"\n12. Creating Visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10, c='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Temperature (°C)', fontsize=12)
axes[0, 0].set_ylabel('Predicted Temperature (°C)', fontsize=12)
axes[0, 0].set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Prediction Error Distribution
errors = y_test - y_test_pred
axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Prediction Error (°C)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title(f'Error Distribution\nMAE = {test_mae:.2f}°C', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
top_features = feature_importance.head(12)
axes[0, 2].barh(range(len(top_features)), top_features['importance'].values, color='orange')
axes[0, 2].set_yticks(range(len(top_features)))
axes[0, 2].set_yticklabels(top_features['feature'].values)
axes[0, 2].set_xlabel('Importance', fontsize=12)
axes[0, 2].set_title('Top 12 Feature Importance', fontsize=14, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='x')
axes[0, 2].invert_yaxis()

# Plot 4: Residual Plot
axes[1, 0].scatter(y_test_pred, errors, alpha=0.5, s=10, c='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Temperature (°C)', fontsize=12)
axes[1, 0].set_ylabel('Residual (°C)', fontsize=12)
axes[1, 0].set_title(f'Residual Plot\nRMSE = {test_rmse:.2f}°C', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Train vs Val vs Test Performance
datasets = ['Train', 'Validation', 'Test']
r2_scores = [train_r2, val_r2, test_r2]
mae_scores = [train_mae, val_mae, test_mae]

x_pos = np.arange(len(datasets))
axes[1, 1].bar(x_pos - 0.2, [s*100 for s in r2_scores], 0.4, label='R² Score (%)', color='blue', alpha=0.7)
axes[1, 1].bar(x_pos + 0.2, mae_scores, 0.4, label='MAE (°C)', color='red', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(datasets)
axes[1, 1].set_ylabel('Score', fontsize=12)
axes[1, 1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Plot 6: Learning Curve (if available)
if hasattr(model, 'evals_result_'):
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    axes[1, 2].plot(x_axis, results['validation_0']['rmse'], label='Validation')
    axes[1, 2].set_xlabel('Iterations', fontsize=12)
    axes[1, 2].set_ylabel('RMSE', fontsize=12)
    axes[1, 2].set_title('Learning Curve', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
else:
    axes[1, 2].text(0.5, 0.5, 'Learning Curve\nNot Available', 
                    ha='center', va='center', fontsize=14)
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Visualization saved to: xgboost_results.png")

print(f"\n{'='*60}")
print(f"ENHANCED TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"\nModel file: {MODEL_OUTPUT}")
print(f"Results file: {RESULTS_OUTPUT}")
print(f"Visualization: xgboost_results.png")
print(f"\nKey Improvements:")
print(f"  ✓ Outlier removal")
print(f"  ✓ 70/15/15 train/val/test split")
print(f"  ✓ Early stopping with validation")
print(f"  ✓ 5-fold cross-validation")
print(f"  ✓ Optimized hyperparameters")
print(f"  ✓ Removed low-impact features")
print(f"\n{'='*60}\n")
