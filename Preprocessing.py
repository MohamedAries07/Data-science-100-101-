"""
CarbonTrack - Data Preprocessing Module
Project: Predicting Household Carbon Footprint Using Lifestyle Data
Author: Mohamed Aries B (RA2311026050100)
Department: CSE - AIML, SRM Institute of Science and Technology
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# 1. Generate Synthetic Dataset
# ─────────────────────────────────────────────────────────
def generate_dataset(n=1500, seed=42):
    """Generate a realistic synthetic household carbon footprint dataset."""
    np.random.seed(seed)

    vehicle_types  = ['EV', 'Hybrid', 'Petrol', 'Diesel', 'None']
    food_habits    = ['Vegan', 'Vegetarian', 'Mixed', 'Meat-heavy']
    income_levels  = ['Low', 'Middle', 'High']

    vehicle_emission = {'EV': 0.0, 'Hybrid': 0.5, 'Petrol': 1.0, 'Diesel': 1.2, 'None': 0.0}
    food_emission    = {'Vegan': 0.3, 'Vegetarian': 0.5, 'Mixed': 0.8, 'Meat-heavy': 1.0}
    income_mult      = {'Low': 0.8, 'Middle': 1.0, 'High': 1.3}

    electricity_kwh   = np.random.normal(320, 80, n).clip(50, 700)
    fuel_liters       = np.random.normal(80, 30, n).clip(0, 200)
    travel_km_day     = np.random.normal(35, 15, n).clip(0, 120)
    water_liters_day  = np.random.normal(200, 60, n).clip(50, 500)
    waste_kg_week     = np.random.normal(12, 4, n).clip(2, 35)
    family_size       = np.random.randint(1, 8, n)

    vehicle_col = np.random.choice(vehicle_types, n, p=[0.08, 0.12, 0.45, 0.25, 0.10])
    food_col    = np.random.choice(food_habits,   n, p=[0.10, 0.20, 0.45, 0.25])
    income_col  = np.random.choice(income_levels, n, p=[0.30, 0.45, 0.25])

    # Compute carbon score with domain-realistic weights + noise
    carbon_score = (
        electricity_kwh  * 0.40 +
        fuel_liters      * 0.80 +
        travel_km_day    * 1.20 +
        water_liters_day * 0.05 +
        waste_kg_week    * 2.50 +
        family_size      * 15.0 +
        np.array([vehicle_emission[v] for v in vehicle_col]) * 120 +
        np.array([food_emission[f]    for f in food_col])    * 80  +
        np.array([income_mult[i]      for i in income_col])  * 40  +
        np.random.normal(0, 20, n)
    ).clip(100, 1200)

    # Introduce ~3% missing values
    df = pd.DataFrame({
        'electricity_kwh':  electricity_kwh,
        'fuel_liters':      fuel_liters,
        'vehicle_type':     vehicle_col,
        'travel_km_day':    travel_km_day,
        'food_habit':       food_col,
        'water_liters_day': water_liters_day,
        'waste_kg_week':    waste_kg_week,
        'family_size':      family_size,
        'income_level':     income_col,
        'carbon_score':     carbon_score
    })

    for col in ['electricity_kwh', 'fuel_liters', 'travel_km_day', 'water_liters_day']:
        mask = np.random.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────────────────
# 2. Handle Missing Values
# ─────────────────────────────────────────────────────────
def handle_missing(df):
    """Impute missing numerical values with column median."""
    print("\n[1] Missing Values Before Imputation:")
    print(df.isnull().sum())

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    ✔ '{col}' filled with median = {median_val:.2f}")

    print("\n[✓] Missing values after imputation:", df.isnull().sum().sum())
    return df


# ─────────────────────────────────────────────────────────
# 3. Encode Categorical Features
# ─────────────────────────────────────────────────────────
def encode_categoricals(df):
    """Label-encode categorical columns and save mappings."""
    cat_cols = ['vehicle_type', 'food_habit', 'income_level']
    encoders = {}

    print("\n[2] Encoding Categorical Features:")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"    ✔ '{col}' → {encoders[col]}")

    return df, encoders


# ─────────────────────────────────────────────────────────
# 4. Feature Scaling
# ─────────────────────────────────────────────────────────
def scale_features(df, target_col='carbon_score'):
    """Apply StandardScaler to all numeric features (excluding target)."""
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("\n[3] Feature Scaling Applied (StandardScaler).")
    return df_scaled, scaler


# ─────────────────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────────────────
def run_preprocessing():
    os.makedirs('dataset/raw_data',       exist_ok=True)
    os.makedirs('dataset/processed_data', exist_ok=True)

    print("=" * 55)
    print("  CarbonTrack — Data Preprocessing Pipeline")
    print("=" * 55)

    # Generate and save raw data
    df_raw = generate_dataset()
    df_raw.to_csv('dataset/raw_data/carbon_footprint_raw.csv', index=False)
    print(f"\n[✓] Raw dataset saved  → {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
    print("\nSample (raw):")
    print(df_raw.head(3).to_string())

    # Preprocessing pipeline
    df = df_raw.copy()
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)
    df_scaled, scaler = scale_features(df)

    # Save processed data
    df.to_csv('dataset/processed_data/carbon_footprint_encoded.csv',  index=False)
    df_scaled.to_csv('dataset/processed_data/carbon_footprint_scaled.csv', index=False)
    print("\n[✓] Encoded dataset saved → dataset/processed_data/carbon_footprint_encoded.csv")
    print("[✓] Scaled dataset saved  → dataset/processed_data/carbon_footprint_scaled.csv")

    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    return df, df_scaled, scaler, encoders


if __name__ == '__main__':
    run_preprocessing()
