"""
End-to-End Training Pipeline for DynaPrice AI.
Generates benchmark data, estimates price elasticity, trains gradient boosting regressor,
evaluates metrics, renders publication-grade diagnostic plots, and serializes artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from src.data_generator import generate_pricing_dataset
from src.models import PriceElasticityAnalyzer, DemandForecaster, DynamicPricingOptimizer


def run_training_pipeline():
    print("=" * 70)
    print(" [DynaPrice AI] Initiating Production Training Pipeline...")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Dataset Generation
    print("--> 1/5 Generating synthetic multi-category retail dataset (12,000 transactions)...")
    df = generate_pricing_dataset(n_samples=12000, random_state=42)
    csv_path = os.path.join(data_dir, "retail_pricing_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"    Saved raw benchmark data to {csv_path} (Shape: {df.shape})")

    # 2. Train-Test Split
    print("--> 2/5 Splitting data (80% train, 20% test)...")
    feature_cols = [
        "category", "offered_price", "list_price", "unit_cost",
        "competitor_price", "inventory_level", "customer_rating",
        "is_weekend", "is_holiday_promo", "ad_spend"
    ]
    target_col = "units_sold"

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    # 3. Model Training
    print("--> 3/5 Fitting Econometric Price Elasticity Model...")
    elasticity_analyzer = PriceElasticityAnalyzer()
    elasticity_analyzer.fit(train_df)

    for cat, info in elasticity_analyzer.category_models.items():
        print(f"    * Category: {cat:<24} | PED: {info['ped']:.2f} | Cross-PED: {info['cross_ped']:.2f} | R2: {info['r2']:.2f}")

    print("--> 4/5 Training Gradient Boosting Demand Regressor...")
    forecaster = DemandForecaster(n_estimators=160, learning_rate=0.08, max_depth=5)
    forecaster.fit(X_train, y_train)

    y_pred = forecaster.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100)

    print("    Benchmark Test Results:")
    print(f"    - R^2 Score   : {r2:.4f}")
    print(f"    - MAE         : {mae:.2f} units")
    print(f"    - RMSE        : {rmse:.2f} units")
    print(f"    - MAPE        : {mape:.2f}%")

    optimizer = DynamicPricingOptimizer(forecaster, elasticity_analyzer)

    # 4. Save Metrics & Model Artifacts
    metrics_payload = {
        "dataset_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "category_elasticities": elasticity_analyzer.category_models,
        "feature_importances": forecaster.get_feature_importances(),
    }
    metrics_path = os.path.join(artifacts_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"    Saved evaluation metrics to {metrics_path}")

    model_payload = {
        "forecaster": forecaster,
        "elasticity_analyzer": elasticity_analyzer,
        "optimizer": optimizer,
        "metrics": metrics_payload,
    }
    model_path = os.path.join(artifacts_dir, "pricing_engine.joblib")
    joblib.dump(model_payload, model_path)
    print(f"    Serialized complete engine bundle to {model_path}")

    # 5. Diagnostic Visualizations
    print("--> 5/5 Generating Diagnostic Plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Optimization Curves (Demand & Profit vs. Price)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_prod = {
        "category": "Consumer Electronics",
        "unit_cost": 85.0,
        "list_price": 149.99,
        "current_offered_price": 139.99,
        "competitor_price": 145.0,
        "inventory_level": 250,
        "customer_rating": 4.5,
    }
    opt_res = optimizer.optimize_price(**sample_prod)
    curve = opt_res["simulation_curve"]

    # Demand curve
    axes[0].plot(curve["prices"], curve["demands"], color="#1E88E5", lw=2.5, label="Expected Units (Demand)")
    axes[0].axvline(opt_res["optimal_price"], color="#D81B60", linestyle="--", label=f"Optimal Price (${opt_res['optimal_price']:.2f})")
    axes[0].axvline(sample_prod["current_offered_price"], color="#757575", linestyle=":", label=f"Current Price (${sample_prod['current_offered_price']:.2f})")
    axes[0].set_title("Price vs. Predicted Sales Volume", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Offered Price ($)")
    axes[0].set_ylabel("Predicted Daily Demand (Units)")
    axes[0].legend()

    # Profit & Revenue curves
    axes[1].plot(curve["prices"], curve["profits"], color="#004D40", lw=2.5, label="Gross Profit ($)")
    axes[1].plot(curve["prices"], curve["revenues"], color="#FFC107", lw=2, linestyle="-.", label="Gross Revenue ($)")
    axes[1].axvline(opt_res["optimal_price"], color="#D81B60", linestyle="--", label=f"Max Profit Equilibrium (${opt_res['optimal_projected_profit']:.2f})")
    axes[1].set_title("Price vs. Financial Yield (Profit & Revenue)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Offered Price ($)")
    axes[1].set_ylabel("Yield ($)")
    axes[1].legend()

    plt.tight_layout()
    curve_fig_path = os.path.join(artifacts_dir, "price_elasticity_demand_curves.png")
    fig.savefig(curve_fig_path, dpi=300)
    plt.close(fig)
    print(f"    Saved optimization curves to {curve_fig_path}")

    # Plot 2: Model Diagnostics (Actual vs Predicted + Feature Importances)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted scatter
    sub_test = y_test[:400]
    sub_pred = y_pred[:400]
    axes2[0].scatter(sub_test, sub_pred, alpha=0.45, color="#2E7D32", edgecolors="none")
    max_val = max(sub_test.max(), sub_pred.max())
    axes2[0].plot([0, max_val], [0, max_val], color="crimson", linestyle="--", lw=1.8, label="Ideal 45° Line")
    axes2[0].set_title(f"Model Calibration: Actual vs Predicted Demand (R²={r2:.3f})", fontsize=12, fontweight="bold")
    axes2[0].set_xlabel("Actual Demand (Units)")
    axes2[0].set_ylabel("Predicted Demand (Units)")
    axes2[0].legend()

    # Feature Importances Bar Chart
    top_feats = list(forecaster.get_feature_importances().items())[:8]
    feats, vals = zip(*reversed(top_feats))
    axes2[1].barh(feats, vals, color="#00838F")
    axes2[1].set_title("Top 8 Demand Predictive Drivers", fontsize=12, fontweight="bold")
    axes2[1].set_xlabel("Relative Importance Score")

    plt.tight_layout()
    diag_fig_path = os.path.join(artifacts_dir, "model_diagnostics.png")
    fig2.savefig(diag_fig_path, dpi=300)
    plt.close(fig2)
    print(f"    Saved model diagnostics to {diag_fig_path}")

    print("=" * 70)
    print(" [DynaPrice AI] Training Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
