"""
Comprehensive Test Suite for DynaPrice AI.
Covers data generation, econometric models, pricing optimizer guardrails, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.data_generator import generate_pricing_dataset, CATEGORIES_CONFIG
from src.features import PricingFeatureEngineer
from src.models import PriceElasticityAnalyzer, DemandForecaster, DynamicPricingOptimizer
from api.main import app, get_engine


@pytest.fixture(scope="session")
def benchmark_data():
    return generate_pricing_dataset(n_samples=500, random_state=42)


@pytest.fixture(scope="session")
def trained_engine(benchmark_data):
    df = benchmark_data
    elasticity_analyzer = PriceElasticityAnalyzer()
    elasticity_analyzer.fit(df)

    feature_cols = [
        "category", "offered_price", "list_price", "unit_cost",
        "competitor_price", "inventory_level", "customer_rating",
        "is_weekend", "is_holiday_promo", "ad_spend"
    ]
    forecaster = DemandForecaster(n_estimators=30, learning_rate=0.1, max_depth=3)
    forecaster.fit(df[feature_cols], df["units_sold"].values)

    optimizer = DynamicPricingOptimizer(forecaster, elasticity_analyzer)
    return {
        "analyzer": elasticity_analyzer,
        "forecaster": forecaster,
        "optimizer": optimizer
    }


def test_data_generation_contract(benchmark_data):
    df = benchmark_data
    assert len(df) == 500
    assert "units_sold" in df.columns
    assert "offered_price" in df.columns
    assert (df["units_sold"] >= 0).all()
    assert (df["offered_price"] > 0).all()
    assert (df["offered_price"] >= df["unit_cost"]).all()


def test_feature_engineering_pipeline(benchmark_data):
    fe = PricingFeatureEngineer()
    transformed = fe.fit_transform(benchmark_data)
    assert "price_ratio_vs_competitor" in transformed.columns
    assert "margin_potential" in transformed.columns
    assert "inventory_scarcity_flag" in transformed.columns
    assert not transformed.isnull().values.any()


def test_price_elasticity_economic_direction(trained_engine):
    analyzer = trained_engine["analyzer"]
    # In standard economics, price elasticity of demand must be negative (downward sloping demand)
    for cat, info in analyzer.category_models.items():
        assert info["ped"] < 0, f"Category {cat} should have negative price elasticity."


def test_pricing_optimizer_guardrails(trained_engine):
    optimizer = trained_engine["optimizer"]
    unit_cost = 60.0
    list_price = 100.0
    current_price = 95.0
    competitor_price = 90.0

    res = optimizer.optimize_price(
        category="Consumer Electronics",
        unit_cost=unit_cost,
        list_price=list_price,
        current_offered_price=current_price,
        competitor_price=competitor_price,
        inventory_level=200,
        min_margin_pct=0.10,
        max_surge_pct=0.20
    )

    opt_p = res["optimal_price"]
    # Must enforce min margin: P >= unit_cost * 1.10 = 66.0
    assert opt_p >= (unit_cost * 1.10) - 0.01, f"Optimal price {opt_p} breached margin floor!"
    # Must enforce surge ceiling: P <= list_price * 1.20 = 120.0
    assert opt_p <= (list_price * 1.20) + 0.01, f"Optimal price {opt_p} breached surge cap!"
    assert res["optimal_expected_units"] >= 0
    assert res["strategy_action"] in ["MARKUP", "MARKDOWN", "MAINTAIN"]


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]


def test_api_optimize_price_endpoint():
    client = TestClient(app)
    payload = {
        "category": "Consumer Electronics",
        "unit_cost": 70.0,
        "list_price": 129.99,
        "current_offered_price": 119.99,
        "competitor_price": 115.00,
        "inventory_level": 150,
        "customer_rating": 4.5,
        "optimization_objective": "profit"
    }
    response = client.post("/optimize-price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert "profit_uplift_pct" in data
    assert "strategy_action" in data
    assert data["optimal_price"] > 70.0


def test_api_predict_demand_endpoint():
    client = TestClient(app)
    payload = {
        "category": "Luxury Apparel",
        "offered_price": 250.0,
        "list_price": 280.0,
        "unit_cost": 90.0,
        "competitor_price": 260.0,
        "inventory_level": 80,
        "customer_rating": 4.8
    }
    response = client.post("/predict-demand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_daily_units"] >= 0
    assert data["estimated_revenue"] >= 0
