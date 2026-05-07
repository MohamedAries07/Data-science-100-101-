import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import run_preprocessing

sns.set_theme(style='whitegrid', font_scale=1.1)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('outputs/graphs',  exist_ok=True)


# ─────────────────────────────────────────────────────────
# 1. Prepare Data
# ─────────────────────────────────────────────────────────
def prepare_data():
    run_preprocessing()
    df = pd.read_csv("dataset/processed_data/carbon_footprint_encoded.csv")
    df.dropna(inplace=True)
    X = df.drop(columns=['carbon_score'])
    y = df['carbon_score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    print(f"\n[✓] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────────────────
# 2. Train Models
# ─────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    print("\n[Training Models...]")

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  ✔ Random Forest Regressor trained.")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("  ✔ Linear Regression trained.")

    return rf, lr


# ─────────────────────────────────────────────────────────
# 3. Evaluate Models
# ─────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  = {mae:.2f}")
    print(f"    RMSE = {rmse:.2f}")
    print(f"    R²   = {r2:.4f}")
    return {'model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)}, y_pred


# ─────────────────────────────────────────────────────────
# 4. Plot: Actual vs Predicted
# ─────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, rf_pred, lr_pred):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    preds = [('Random Forest', rf_pred, '#2E86AB'), ('Linear Regression', lr_pred, '#A23B72')]
    for i, (name, y_pred, color) in enumerate(preds):
        axes[i].scatter(y_test, y_pred, alpha=0.4, color=color, s=18, label='Predictions')
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[i].plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect Fit')
        axes[i].set_title(f'{name}: Actual vs Predicted', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Actual Carbon Score')
        axes[i].set_ylabel('Predicted Carbon Score')
        axes[i].legend()
        r2 = r2_score(y_test, y_pred)
        axes[i].annotate(f'R² = {r2:.4f}', xy=(0.05, 0.92), xycoords='axes fraction',
                         fontsize=12, color='black',
                         bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
    plt.suptitle('Model Performance: Actual vs Predicted Carbon Scores', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('outputs/graphs/08_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [✓] Saved → outputs/graphs/08_actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────
# 5. Plot: Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.set_title('Feature Importance — Random Forest Regressor', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.set_xlim(0, importances.max() + 0.06)
    plt.tight_layout()
    fig.savefig('outputs/graphs/09_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [✓] Saved → outputs/graphs/09_feature_importance.png")


# ─────────────────────────────────────────────────────────
# 6. Plot: Model Comparison Bar
# ─────────────────────────────────────────────────────────
def plot_model_comparison(rf_metrics, lr_metrics):
    metrics = ['MAE', 'RMSE', 'R2']
    rf_vals = [rf_metrics['MAE'], rf_metrics['RMSE'], rf_metrics['R2'] * 100]
    lr_vals = [lr_metrics['MAE'], lr_metrics['RMSE'], lr_metrics['R2'] * 100]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, rf_vals, width, label='Random Forest', color='#2E86AB', alpha=0.85)
    b2 = ax.bar(x + width / 2, lr_vals, width, label='Linear Regression', color='#A23B72', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['MAE (↓ better)', 'RMSE (↓ better)', 'R² × 100 (↑ better)'], fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=11)
    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=10)
    plt.tight_layout()
    fig.savefig('outputs/graphs/10_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [✓] Saved → outputs/graphs/10_model_comparison.png")


# ─────────────────────────────────────────────────────────
# 7. Sample Prediction
# ─────────────────────────────────────────────────────────
def sample_prediction(rf_model, feature_names):
    # vehicle_type: Petrol=3, food_habit: Mixed=2, income_level: Middle=1
    sample = pd.DataFrame([{
        'electricity_kwh':  320,
        'fuel_liters':       85,
        'vehicle_type':       3,
        'travel_km_day':     40,
        'food_habit':         2,
        'water_liters_day': 200,
        'waste_kg_week':     12,
        'family_size':        4,
        'income_level':       1
    }], columns=feature_names)
    prediction = rf_model.predict(sample)[0]
    print(f"\n  [Sample Prediction]")
    print(f"    Input  : Electricity=320 kWh, Fuel=85 L, Vehicle=Petrol, Travel=40 km/day")
    print(f"             Food=Mixed, Water=200 L/day, Waste=12 kg/wk, Family=4, Income=Middle")
    print(f"    Output : Predicted Carbon Score = {prediction:.2f} kg CO₂/month")
    return prediction


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def run_model():
    print("=" * 55)
    print("  CarbonTrack — Model Training & Evaluation")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    rf_model, lr_model = train_models(X_train, y_train)

    print("\n[Evaluating Models...]")
    rf_metrics, rf_pred = evaluate(rf_model, X_test, y_test, "Random Forest Regressor")
    lr_metrics, lr_pred = evaluate(lr_model, X_test, y_test, "Linear Regression")

    print("\n[Generating Plots...]")
    plot_actual_vs_predicted(y_test, rf_pred, lr_pred)
    plot_feature_importance(rf_model, feature_names)
    plot_model_comparison(rf_metrics, lr_metrics)

    sample_prediction(rf_model, feature_names)

    # Save metrics
    results = {'RandomForest': rf_metrics, 'LinearRegression': lr_metrics}
    with open('outputs/results/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  [✓] Metrics saved → outputs/results/model_metrics.json")

    print("\n" + "=" * 55)
    print("  Model Pipeline Complete!")
    print("=" * 55)


if __name__ == '__main__':
    run_model()
