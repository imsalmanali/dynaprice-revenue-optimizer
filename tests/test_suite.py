"""
Standard Library Test Suite for DynaPrice AI.
Runs seamlessly with python -m unittest.
"""

import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.data_generator import generate_pricing_dataset
from src.features import PricingFeatureEngineer
from src.models import PriceElasticityAnalyzer, DemandForecaster, DynamicPricingOptimizer
from api.main import app, get_engine


class TestDynaPriceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = generate_pricing_dataset(n_samples=500, random_state=42)
        cls.elasticity_analyzer = PriceElasticityAnalyzer()
        cls.elasticity_analyzer.fit(cls.df)

        feature_cols = [
            "category", "offered_price", "list_price", "unit_cost",
            "competitor_price", "inventory_level", "customer_rating",
            "is_weekend", "is_holiday_promo", "ad_spend"
        ]
        cls.forecaster = DemandForecaster(n_estimators=30, learning_rate=0.1, max_depth=3)
        cls.forecaster.fit(cls.df[feature_cols], cls.df["units_sold"].values)
        cls.optimizer = DynamicPricingOptimizer(cls.forecaster, cls.elasticity_analyzer)
        cls.client = TestClient(app)

    def test_01_data_generation_contract(self):
        self.assertEqual(len(self.df), 500)
        self.assertIn("units_sold", self.df.columns)
        self.assertIn("offered_price", self.df.columns)
        self.assertTrue((self.df["units_sold"] >= 0).all())
        self.assertTrue((self.df["offered_price"] > 0).all())
        self.assertTrue((self.df["offered_price"] >= self.df["unit_cost"]).all())

    def test_02_feature_engineering_pipeline(self):
        fe = PricingFeatureEngineer()
        transformed = fe.fit_transform(self.df)
        self.assertIn("price_ratio_vs_competitor", transformed.columns)
        self.assertIn("margin_potential", transformed.columns)
        self.assertIn("inventory_scarcity_flag", transformed.columns)
        self.assertFalse(transformed.isnull().values.any())

    def test_03_price_elasticity_economic_direction(self):
        # Microeconomic theory: PED must be negative (downward sloping demand)
        for cat, info in self.elasticity_analyzer.category_models.items():
            self.assertLess(info["ped"], 0, f"Category {cat} should have negative price elasticity.")

    def test_04_pricing_optimizer_guardrails(self):
        unit_cost = 60.0
        list_price = 100.0
        current_price = 95.0
        competitor_price = 90.0

        res = self.optimizer.optimize_price(
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
        self.assertGreaterEqual(opt_p, (unit_cost * 1.10) - 0.05)
        self.assertLessEqual(opt_p, (list_price * 1.20) + 0.05)
        self.assertGreaterEqual(res["optimal_expected_units"], 0)
        self.assertIn(res["strategy_action"], ["MARKUP", "MARKDOWN", "MAINTAIN"])

    def test_05_api_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["healthy", "degraded"])

    def test_06_api_optimize_price_endpoint(self):
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
        response = self.client.post("/optimize-price", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("optimal_price", data)
        self.assertIn("profit_uplift_pct", data)
        self.assertGreater(data["optimal_price"], 70.0)

    def test_07_api_predict_demand_endpoint(self):
        payload = {
            "category": "Luxury Apparel",
            "offered_price": 250.0,
            "list_price": 280.0,
            "unit_cost": 90.0,
            "competitor_price": 260.0,
            "inventory_level": 80,
            "customer_rating": 4.8
        }
        response = self.client.post("/predict-demand", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["predicted_daily_units"], 0)
        self.assertGreaterEqual(data["estimated_revenue"], 0)


if __name__ == "__main__":
    unittest.main()
