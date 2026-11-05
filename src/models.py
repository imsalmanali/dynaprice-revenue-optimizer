"""
Machine Learning & Econometric Dynamic Pricing Models.
Contains:
1. Log-Log Econometric Price Elasticity of Demand (PED) Estimator.
2. Gradient Boosting Demand Regressor.
3. Revenue & Profit Grid-Search Optimization Engine with Business Guardrails.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.features import PricingFeatureEngineer


class PriceElasticityAnalyzer:
    """
    Econometric Log-Log Regression Model:
    log(Quantity) = alpha + beta * log(Price) + gamma * log(CompetitorPrice)
    where beta represents the Price Elasticity of Demand (PED).
    """

    def __init__(self):
        self.category_models: Dict[str, Dict[str, Any]] = {}
        self.global_elasticity: float = -1.5

    def fit(self, df: pd.DataFrame):
        valid = df[(df["units_sold"] > 0) & (df["offered_price"] > 0)].copy()

        # Fit category-specific elasticity
        for cat in valid["category"].unique():
            cat_df = valid[valid["category"] == cat]
            if len(cat_df) < 20:
                continue

            X_cat = np.column_stack([
                np.log(cat_df["offered_price"].values),
                np.log(cat_df["competitor_price"].values)
            ])
            y_cat = np.log(cat_df["units_sold"].values)

            reg = LinearRegression().fit(X_cat, y_cat)
            ped = float(reg.coef_[0])
            cross_ped = float(reg.coef_[1])

            self.category_models[cat] = {
                "ped": ped,
                "cross_ped": cross_ped,
                "intercept": float(reg.intercept_),
                "r2": float(reg.score(X_cat, y_cat)),
                "elasticity_class": self.classify_elasticity(ped)
            }

        # Global PED fallback
        X_all = np.log(valid["offered_price"].values).reshape(-1, 1)
        y_all = np.log(valid["units_sold"].values)
        reg_all = LinearRegression().fit(X_all, y_all)
        self.global_elasticity = float(reg_all.coef_[0])

    @staticmethod
    def classify_elasticity(ped: float) -> str:
        abs_ped = abs(ped)
        if abs_ped > 1.25:
            return "Highly Elastic (Sensitive to Price Increases)"
        elif abs_ped < 0.85:
            return "Inelastic (Pricing Power / High Margin Opportunity)"
        else:
            return "Unitary Elastic (Balanced Volume/Price Tradeoff)"

    def get_category_metrics(self, category: str) -> Dict[str, Any]:
        return self.category_models.get(category, {
            "ped": self.global_elasticity,
            "cross_ped": 0.5,
            "r2": 0.5,
            "elasticity_class": self.classify_elasticity(self.global_elasticity)
        })


class DemandForecaster:
    """
    Non-linear Gradient Boosting Regressor predicting expected units sold
    from dynamic price signals, inventory, ratings, and market context.
    """

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.08, max_depth: int = 5):
        self.feature_engineer = PricingFeatureEngineer()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_trans = self.feature_engineer.fit_transform(X)
        self.model.fit(X_trans, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_trans = self.feature_engineer.transform(X)
        preds = self.model.predict(X_trans)
        return np.clip(preds, 0, None)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        names = self.feature_engineer.feature_names_
        imps = self.model.feature_importances_
        sorted_pairs = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in sorted_pairs}


class DynamicPricingOptimizer:
    """
    Prescriptive Optimization Engine:
    Simulates demand across candidate pricing grids to compute optimal prices
    maximizing gross profit or gross revenue, while applying strict business rules.
    """

    def __init__(self, forecaster: DemandForecaster, elasticity_analyzer: PriceElasticityAnalyzer):
        self.forecaster = forecaster
        self.elasticity_analyzer = elasticity_analyzer

    def optimize_price(
        self,
        category: str,
        unit_cost: float,
        list_price: float,
        current_offered_price: float,
        competitor_price: float,
        inventory_level: int,
        customer_rating: float = 4.2,
        is_weekend: int = 0,
        is_holiday_promo: int = 0,
        ad_spend: float = 25.0,
        optimization_objective: str = "profit",  # "profit" or "revenue"
        min_margin_pct: float = 0.08,            # business guardrail: at least 8% margin
        max_surge_pct: float = 0.25,             # business guardrail: max +25% above MSRP
        price_steps: int = 40,
    ) -> Dict[str, Any]:
        """
        Evaluate price grid, compute demand curve, and pinpoint the optimal price point.
        """
        # Determine pricing candidate boundary
        min_allowed_price = max(round(unit_cost * (1.0 + min_margin_pct), 2), round(list_price * 0.65, 2))
        max_allowed_price = round(list_price * (1.0 + max_surge_pct), 2)

        if min_allowed_price >= max_allowed_price:
            max_allowed_price = min_allowed_price * 1.25

        candidate_prices = np.linspace(min_allowed_price, max_allowed_price, price_steps)

        # Build feature DataFrame across all candidate price points
        grid_rows = []
        for p in candidate_prices:
            grid_rows.append({
                "category": category,
                "offered_price": float(p),
                "list_price": list_price,
                "unit_cost": unit_cost,
                "competitor_price": competitor_price,
                "inventory_level": inventory_level,
                "customer_rating": customer_rating,
                "is_weekend": is_weekend,
                "is_holiday_promo": is_holiday_promo,
                "ad_spend": ad_spend,
            })

        df_grid = pd.DataFrame(grid_rows)
        predicted_demands = self.forecaster.predict(df_grid)
        # Demand physically capped by available inventory
        predicted_demands = np.minimum(predicted_demands, inventory_level)

        revenues = candidate_prices * predicted_demands
        profits = (candidate_prices - unit_cost) * predicted_demands

        # Select optimum index
        if optimization_objective == "profit":
            opt_idx = int(np.argmax(profits))
        else:
            opt_idx = int(np.argmax(revenues))

        opt_price = float(np.round(candidate_prices[opt_idx], 2))
        opt_demand = float(np.round(predicted_demands[opt_idx], 1))
        opt_revenue = float(np.round(revenues[opt_idx], 2))
        opt_profit = float(np.round(profits[opt_idx], 2))
        opt_margin_pct = float(np.round((opt_profit / opt_revenue * 100) if opt_revenue > 0 else 0.0, 2))

        # Baseline evaluation at current price
        curr_row = pd.DataFrame([{
            "category": category,
            "offered_price": float(current_offered_price),
            "list_price": list_price,
            "unit_cost": unit_cost,
            "competitor_price": competitor_price,
            "inventory_level": inventory_level,
            "customer_rating": customer_rating,
            "is_weekend": is_weekend,
            "is_holiday_promo": is_holiday_promo,
            "ad_spend": ad_spend,
        }])
        curr_demand = float(np.round(min(self.forecaster.predict(curr_row)[0], inventory_level), 1))
        curr_revenue = float(np.round(curr_demand * current_offered_price, 2))
        curr_profit = float(np.round(curr_demand * (current_offered_price - unit_cost), 2))
        curr_margin_pct = float(np.round((curr_profit / curr_revenue * 100) if curr_revenue > 0 else 0.0, 2))

        profit_uplift_pct = float(np.round(
            ((opt_profit - curr_profit) / curr_profit * 100) if curr_profit > 0 else 0.0, 2
        ))
        revenue_uplift_pct = float(np.round(
            ((opt_revenue - curr_revenue) / curr_revenue * 100) if curr_revenue > 0 else 0.0, 2
        ))

        # Strategic Action Recommendation
        price_diff = opt_price - current_offered_price
        if abs(price_diff) < 0.5:
            strategy_action = "MAINTAIN"
            strategy_reason = "Current price is already within 0.5% of optimal profit equilibrium."
        elif price_diff > 0:
            strategy_action = "MARKUP"
            strategy_reason = f"Increase price by ${price_diff:.2f} (+{(price_diff/current_offered_price)*100:.1f}%). Low demand sensitivity unlocks higher margin."
        else:
            strategy_action = "MARKDOWN"
            strategy_reason = f"Discount price by ${abs(price_diff):.2f} ({(price_diff/current_offered_price)*100:.1f}%). Higher price elasticity yields volume surge and profit expansion."

        elasticity_info = self.elasticity_analyzer.get_category_metrics(category)

        return {
            "optimal_price": opt_price,
            "optimal_expected_units": opt_demand,
            "optimal_projected_revenue": opt_revenue,
            "optimal_projected_profit": opt_profit,
            "optimal_margin_pct": opt_margin_pct,
            "current_price": current_offered_price,
            "current_expected_units": curr_demand,
            "current_projected_revenue": curr_revenue,
            "current_projected_profit": curr_profit,
            "current_margin_pct": curr_margin_pct,
            "profit_uplift_pct": profit_uplift_pct,
            "revenue_uplift_pct": revenue_uplift_pct,
            "strategy_action": strategy_action,
            "strategy_reason": strategy_reason,
            "elasticity_metrics": elasticity_info,
            "guardrails": {
                "min_allowed_price": min_allowed_price,
                "max_allowed_price": max_allowed_price,
                "min_margin_pct": min_margin_pct,
                "max_surge_pct": max_surge_pct,
            },
            "simulation_curve": {
                "prices": [float(p) for p in np.round(candidate_prices, 2)],
                "demands": [float(d) for d in np.round(predicted_demands, 1)],
                "revenues": [float(r) for r in np.round(revenues, 2)],
                "profits": [float(pr) for pr in np.round(profits, 2)],
            }
        }
