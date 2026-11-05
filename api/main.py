"""
FastAPI Microservice for DynaPrice AI.
Provides real-time dynamic pricing optimization, elasticity scoring, and demand inference.
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

from api.schemas import (
    PriceOptimizationRequest,
    PriceOptimizationResponse,
    DemandPredictionRequest,
    DemandPredictionResponse,
    BatchRepriceRequest,
    BatchRepriceResponse,
    ElasticityInfo,
    GuardrailsInfo,
    SimulationCurve,
)
from src.models import DynamicPricingOptimizer, DemandForecaster, PriceElasticityAnalyzer

app = FastAPI(
    title="DynaPrice AI - Dynamic Pricing Engine API",
    description="Production Machine Learning & Microeconomic Optimization Microservice for Real-Time Price Optimization and Demand Forecasting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: Optional[Dict[str, Any]] = None


def get_engine() -> Dict[str, Any]:
    global ENGINE
    if ENGINE is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        art_path = os.path.join(base_dir, "artifacts", "pricing_engine.joblib")
        if not os.path.exists(art_path):
            raise RuntimeError(f"Model artifacts not found at {art_path}. Run 'python src/train.py' first.")
        ENGINE = joblib.load(art_path)
    return ENGINE


@app.on_event("startup")
def startup_event():
    try:
        get_engine()
        print("[DynaPrice API] Model artifacts successfully loaded into memory.")
    except Exception as e:
        print(f"[DynaPrice API] Notice: Artifacts pending generation: {e}")


@app.get("/", tags=["General"])
def root():
    return {
        "service": "DynaPrice AI",
        "description": "Enterprise Dynamic Pricing & Demand Forecasting Engine",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "optimize_price": "/optimize-price",
            "predict_demand": "/predict-demand",
            "batch_reprice": "/batch-reprice",
            "metrics": "/metrics",
            "docs": "/docs",
        }
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    try:
        eng = get_engine()
        ready = eng is not None and "optimizer" in eng
    except Exception:
        ready = False
    return {
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "engine_version": "1.0.0"
    }


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    eng = get_engine()
    return eng.get("metrics", {})


@app.post("/optimize-price", response_model=PriceOptimizationResponse, tags=["Pricing Optimization"])
def optimize_price(req: PriceOptimizationRequest):
    try:
        eng = get_engine()
        optimizer: DynamicPricingOptimizer = eng["optimizer"]

        res = optimizer.optimize_price(
            category=req.category,
            unit_cost=req.unit_cost,
            list_price=req.list_price,
            current_offered_price=req.current_offered_price,
            competitor_price=req.competitor_price,
            inventory_level=req.inventory_level,
            customer_rating=req.customer_rating,
            is_weekend=req.is_weekend,
            is_holiday_promo=req.is_holiday_promo,
            ad_spend=req.ad_spend,
            optimization_objective=req.optimization_objective,
            min_margin_pct=req.min_margin_pct,
            max_surge_pct=req.max_surge_pct,
        )

        return PriceOptimizationResponse(
            optimal_price=res["optimal_price"],
            optimal_expected_units=res["optimal_expected_units"],
            optimal_projected_revenue=res["optimal_projected_revenue"],
            optimal_projected_profit=res["optimal_projected_profit"],
            optimal_margin_pct=res["optimal_margin_pct"],
            current_price=res["current_price"],
            current_expected_units=res["current_expected_units"],
            current_projected_revenue=res["current_projected_revenue"],
            current_projected_profit=res["current_projected_profit"],
            current_margin_pct=res["current_margin_pct"],
            profit_uplift_pct=res["profit_uplift_pct"],
            revenue_uplift_pct=res["revenue_uplift_pct"],
            strategy_action=res["strategy_action"],
            strategy_reason=res["strategy_reason"],
            elasticity_metrics=ElasticityInfo(**res["elasticity_metrics"]),
            guardrails=GuardrailsInfo(**res["guardrails"]),
            simulation_curve=SimulationCurve(**res["simulation_curve"]),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.post("/predict-demand", response_model=DemandPredictionResponse, tags=["Demand Forecasting"])
def predict_demand(req: DemandPredictionRequest):
    try:
        eng = get_engine()
        forecaster: DemandForecaster = eng["forecaster"]

        row = pd.DataFrame([{
            "category": req.category,
            "offered_price": req.offered_price,
            "list_price": req.list_price,
            "unit_cost": req.unit_cost,
            "competitor_price": req.competitor_price,
            "inventory_level": req.inventory_level,
            "customer_rating": req.customer_rating,
            "is_weekend": req.is_weekend,
            "is_holiday_promo": req.is_holiday_promo,
            "ad_spend": req.ad_spend,
        }])

        pred_units = float(np.round(min(forecaster.predict(row)[0], req.inventory_level), 1))
        est_rev = float(np.round(pred_units * req.offered_price, 2))
        est_profit = float(np.round(pred_units * (req.offered_price - req.unit_cost), 2))

        return DemandPredictionResponse(
            category=req.category,
            offered_price=req.offered_price,
            predicted_daily_units=pred_units,
            estimated_revenue=est_rev,
            estimated_gross_profit=est_profit,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/batch-reprice", response_model=BatchRepriceResponse, tags=["Pricing Optimization"])
def batch_reprice(batch: BatchRepriceRequest):
    results = []
    total_uplift = 0.0

    for item in batch.items:
        res = optimize_price(item)
        results.append(res)
        total_uplift += max(0.0, res.optimal_projected_profit - res.current_projected_profit)

    return BatchRepriceResponse(
        count=len(results),
        total_projected_profit_uplift=round(total_uplift, 2),
        recommendations=results,
    )
