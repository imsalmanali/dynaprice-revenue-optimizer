"""
Pydantic v2 Request & Response Data Contracts for DynaPrice AI.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PriceOptimizationRequest(BaseModel):
    category: str = Field(
        ...,
        description="Product segment/category",
        examples=["Consumer Electronics"]
    )
    unit_cost: float = Field(
        ...,
        gt=0,
        description="Unit cost of goods sold (COGS) in USD",
        examples=[85.0]
    )
    list_price: float = Field(
        ...,
        gt=0,
        description="MSRP or baseline reference price in USD",
        examples=[149.99]
    )
    current_offered_price: float = Field(
        ...,
        gt=0,
        description="Currently active selling price in USD",
        examples=[139.99]
    )
    competitor_price: float = Field(
        ...,
        gt=0,
        description="Observed market competitor price in USD",
        examples=[145.00]
    )
    inventory_level: int = Field(
        ...,
        ge=0,
        description="Available inventory stock count",
        examples=[250]
    )
    customer_rating: float = Field(
        default=4.3,
        ge=1.0,
        le=5.0,
        description="Product review rating (1.0 to 5.0)"
    )
    is_weekend: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary flag: 1 if weekend, else 0"
    )
    is_holiday_promo: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary flag: 1 if promotional period / holiday, else 0"
    )
    ad_spend: float = Field(
        default=25.0,
        ge=0.0,
        description="Daily advertising spend attributed to product in USD"
    )
    optimization_objective: str = Field(
        default="profit",
        description="Optimization objective: 'profit' or 'revenue'"
    )
    min_margin_pct: float = Field(
        default=0.08,
        ge=0.0,
        le=0.80,
        description="Minimum mandatory gross profit margin ratio guardrail"
    )
    max_surge_pct: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Maximum allowed price surge ratio above MSRP"
    )


class ElasticityInfo(BaseModel):
    ped: float
    cross_ped: float
    r2: float
    elasticity_class: str


class GuardrailsInfo(BaseModel):
    min_allowed_price: float
    max_allowed_price: float
    min_margin_pct: float
    max_surge_pct: float


class SimulationCurve(BaseModel):
    prices: List[float]
    demands: List[float]
    revenues: List[float]
    profits: List[float]


class PriceOptimizationResponse(BaseModel):
    optimal_price: float
    optimal_expected_units: float
    optimal_projected_revenue: float
    optimal_projected_profit: float
    optimal_margin_pct: float
    current_price: float
    current_expected_units: float
    current_projected_revenue: float
    current_projected_profit: float
    current_margin_pct: float
    profit_uplift_pct: float
    revenue_uplift_pct: float
    strategy_action: str
    strategy_reason: str
    elasticity_metrics: ElasticityInfo
    guardrails: GuardrailsInfo
    simulation_curve: Optional[SimulationCurve] = None


class BatchRepriceRequest(BaseModel):
    items: List[PriceOptimizationRequest]


class BatchRepriceResponse(BaseModel):
    count: int
    total_projected_profit_uplift: float
    recommendations: List[PriceOptimizationResponse]


class DemandPredictionRequest(BaseModel):
    category: str
    offered_price: float
    list_price: float
    unit_cost: float
    competitor_price: float
    inventory_level: int
    customer_rating: float = 4.2
    is_weekend: int = 0
    is_holiday_promo: int = 0
    ad_spend: float = 25.0


class DemandPredictionResponse(BaseModel):
    category: str
    offered_price: float
    predicted_daily_units: float
    estimated_revenue: float
    estimated_gross_profit: float
