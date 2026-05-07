"""
CarbonTrack - Exploratory Data Analysis & Visualisation Module
Project: Predicting Household Carbon Footprint Using Lifestyle Data
Author: Mohamed Aries B (RA2311026050100)
Department: CSE - AIML, SRM Institute of Science and Technology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import run_preprocessing

# Aesthetics
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
          '#44BBA4', '#E94F37', '#393E41', '#F5A623', '#7BC67E']
os.makedirs('outputs/graphs', exist_ok=True)


def save(fig, name):
    path = f'outputs/graphs/{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


# ─────────────────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────────────────
def dataset_summary(df_raw):
    print("\n" + "=" * 55)
    print("  1. DATASET SUMMARY")
    print("=" * 55)
    print(f"  Shape       : {df_raw.shape}")
    print(f"  Columns     : {list(df_raw.columns)}")
    print(f"  Dtypes      :\n{df_raw.dtypes.to_string()}")
    print(f"\n  Descriptive Statistics:\n{df_raw.describe().round(2).to_string()}")


# ─────────────────────────────────────────────────────────
# 2. Distribution Histograms
# ─────────────────────────────────────────────────────────
def plot_distributions(df_raw):
    num_cols = ['electricity_kwh', 'fuel_liters', 'travel_km_day',
                'water_liters_day', 'waste_kg_week', 'family_size', 'carbon_score']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df_raw[col].dropna(), bins=30, color=COLORS[i % len(COLORS)],
                     edgecolor='white', alpha=0.85)
        axes[i].set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    axes[-1].set_visible(False)
    fig.suptitle('Feature Distributions — CarbonTrack Dataset', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '01_feature_distributions')


# ─────────────────────────────────────────────────────────
# 3. Correlation Heatmap
# ─────────────────────────────────────────────────────────
def plot_correlation_heatmap(df_enc):
    corr = df_enc.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax,
                annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap — All Features', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    save(fig, '02_correlation_heatmap')


# ─────────────────────────────────────────────────────────
# 4. Carbon Score vs. Key Numerical Features (Scatter)
# ─────────────────────────────────────────────────────────
def plot_scatter_vs_carbon(df_raw):
    num_feats = ['electricity_kwh', 'fuel_liters', 'travel_km_day', 'waste_kg_week']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, feat in enumerate(num_feats):
        axes[i].scatter(df_raw[feat], df_raw['carbon_score'],
                        alpha=0.35, c=COLORS[i], s=15)
        # Trend line
        z = np.polyfit(df_raw[feat].dropna(), df_raw.loc[df_raw[feat].notna(), 'carbon_score'], 1)
        p = np.poly1d(z)
        xp = np.linspace(df_raw[feat].min(), df_raw[feat].max(), 200)
        axes[i].plot(xp, p(xp), 'k--', linewidth=1.5)
        axes[i].set_xlabel(feat.replace('_', ' ').title(), fontsize=11)
        axes[i].set_ylabel('Carbon Score (kg CO₂/month)' if i == 0 else '')
        axes[i].set_title(f'{feat.replace("_", " ").title()} vs Carbon Score', fontsize=11, fontweight='bold')
    fig.suptitle('Numerical Features vs Carbon Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, '03_scatter_vs_carbon')


# ─────────────────────────────────────────────────────────
# 5. Categorical Feature Bar Charts
# ─────────────────────────────────────────────────────────
def plot_categorical_bars(df_raw):
    cat_cols = ['vehicle_type', 'food_habit', 'income_level']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, col in enumerate(cat_cols):
        means = df_raw.groupby(col)['carbon_score'].mean().sort_values()
        axes[i].barh(means.index, means.values, color=COLORS[i * 2: i * 2 + len(means)], edgecolor='white')
        axes[i].set_title(f'Avg Carbon Score by {col.replace("_", " ").title()}',
                          fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Avg Carbon Score (kg CO₂/month)')
        for j, v in enumerate(means.values):
            axes[i].text(v + 3, j, f'{v:.0f}', va='center', fontsize=10)
    fig.suptitle('Carbon Score by Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, '04_categorical_bar_charts')


# ─────────────────────────────────────────────────────────
# 6. Pie Charts — Category Distributions
# ─────────────────────────────────────────────────────────
def plot_pie_charts(df_raw):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pie_cols = ['vehicle_type', 'food_habit', 'income_level']
    for i, col in enumerate(pie_cols):
        counts = df_raw[col].value_counts()
        axes[i].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                    colors=COLORS[:len(counts)], startangle=140,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        axes[i].set_title(f'{col.replace("_", " ").title()} Distribution',
                          fontsize=12, fontweight='bold')
    fig.suptitle('Household Category Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, '05_pie_distributions')


# ─────────────────────────────────────────────────────────
# 7. Carbon Score Box Plot by Vehicle Type
# ─────────────────────────────────────────────────────────
def plot_boxplot_vehicle(df_raw):
    order = df_raw.groupby('vehicle_type')['carbon_score'].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_raw, x='vehicle_type', y='carbon_score',
                order=order, palette='Set2', ax=ax,
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})
    ax.set_title('Carbon Score Distribution by Vehicle Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Vehicle Type')
    ax.set_ylabel('Carbon Score (kg CO₂/month)')
    plt.tight_layout()
    save(fig, '06_boxplot_vehicle_type')


# ─────────────────────────────────────────────────────────
# 8. Carbon Score Distribution (Target Variable)
# ─────────────────────────────────────────────────────────
def plot_target_distribution(df_raw):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_raw['carbon_score'], bins=40, color='#2E86AB',
            edgecolor='white', alpha=0.85, density=True)
    from scipy.stats import norm
    mu, std = df_raw['carbon_score'].mean(), df_raw['carbon_score'].std()
    xmin, xmax = df_raw['carbon_score'].min(), df_raw['carbon_score'].max()
    x = np.linspace(xmin, xmax, 300)
    ax.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2.5, label=f'Normal fit (μ={mu:.0f}, σ={std:.0f})')
    ax.axvline(mu, color='darkred', linestyle='--', linewidth=1.5, label=f'Mean = {mu:.0f}')
    ax.set_title('Target Variable: Carbon Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Carbon Score (kg CO₂/month)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    save(fig, '07_target_distribution')


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def run_analysis():
    print("=" * 55)
    print("  CarbonTrack — EDA & Visualisation Pipeline")
    print("=" * 55)

    df_raw, df_enc, scaler, encoders = run_preprocessing()
    # Re-read raw for categorical EDA
    df_raw_csv = pd.read_csv('dataset/raw_data/carbon_footprint_raw.csv')
    df_enc_csv  = pd.read_csv('dataset/processed_data/carbon_footprint_encoded.csv')

    dataset_summary(df_raw_csv)

    print("\n[Generating visualisations...]")
    plot_distributions(df_raw_csv)
    plot_correlation_heatmap(df_enc_csv)
    plot_scatter_vs_carbon(df_raw_csv)
    plot_categorical_bars(df_raw_csv)
    plot_pie_charts(df_raw_csv)
    plot_boxplot_vehicle(df_raw_csv)
    plot_target_distribution(df_raw_csv)

    print("\n" + "=" * 55)
    print("  EDA Complete! Graphs saved to outputs/graphs/")
    print("=" * 55)


if __name__ == '__main__':
    run_analysis()
