"""
Feature engineering pipeline for Dynamic Pricing and Demand Forecasting.
Extracts microeconomic price signals, competitor benchmark indices, and promotional features.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUMERIC_FEATURES: List[str] = [
    "offered_price",
    "list_price",
    "unit_cost",
    "competitor_price",
    "price_ratio_vs_list",
    "price_ratio_vs_competitor",
    "price_diff_vs_competitor",
    "discount_pct",
    "margin_potential",
    "inventory_level",
    "customer_rating",
    "is_weekend",
    "is_holiday_promo",
    "ad_spend_log",
    "inventory_scarcity_flag",
]

CATEGORICAL_FEATURES: List[str] = [
    "category",
]


class PricingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw retail transaction records into rich econometric features
    for demand regression and elasticity estimation.
    """

    def __init__(self):
        self.category_columns_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        df = self._compute_derivatives(X.copy())
        # Track dummy columns for consistency in inference
        cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
        self.category_columns_ = list(cat_dummies.columns)
        self.feature_names_ = NUMERIC_FEATURES + self.category_columns_
        return self

    def _compute_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle missing or zero checks safely
        safe_list = df["list_price"].replace(0, np.nan).fillna(df["offered_price"])
        safe_comp = df["competitor_price"].replace(0, np.nan).fillna(df["offered_price"])

        df["price_ratio_vs_list"] = df["offered_price"] / safe_list
        df["price_ratio_vs_competitor"] = df["offered_price"] / safe_comp
        df["price_diff_vs_competitor"] = df["offered_price"] - df["competitor_price"]

        if "discount_pct" not in df.columns or df["discount_pct"].isnull().any():
            df["discount_pct"] = np.clip(
                (df["list_price"] - df["offered_price"]) / safe_list * 100.0, 0.0, 100.0
            )

        df["margin_potential"] = np.clip(
            (df["offered_price"] - df["unit_cost"]) / np.maximum(df["offered_price"], 1e-4),
            -1.0, 1.0
        )
        df["ad_spend_log"] = np.log1p(df["ad_spend"].clip(lower=0.0))
        df["inventory_scarcity_flag"] = (df["inventory_level"] < 20).astype(int)

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_derivatives(X.copy())

        # Categorical one-hot encoding aligned with fitted schema
        cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
        for col in self.category_columns_:
            if col not in cat_dummies.columns:
                cat_dummies[col] = 0

        cat_dummies = cat_dummies[self.category_columns_]

        X_processed = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
        return X_processed[self.feature_names_]


def extract_econometric_log_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs log(Q) = alpha + beta * log(P) matrix for econometric elasticity regression.
    Filters out 0 quantity rows to avoid log(0) singularity.
    """
    valid = df[df["units_sold"] > 0].copy()
    log_q = np.log(valid["units_sold"].values)
    log_p = np.log(valid["offered_price"].values)
    return log_p, log_q
