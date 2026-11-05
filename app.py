"""
DynaPrice AI - Streamlit Executive Cockpit & Pricing Simulation Playground.
Interactive price elasticity analysis, profit-maximization grid search, and catalog repricing.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(
    page_title="DynaPrice AI | Dynamic Pricing & Demand Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #004D40);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #546E7A;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .rec-badge-markup {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .rec-badge-markdown {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .rec-badge-maintain {
        background-color: #ECEFF1;
        color: #455A64;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    art_path = os.path.join(base_dir, "artifacts", "pricing_engine.joblib")
    metrics_path = os.path.join(base_dir, "artifacts", "benchmark_metrics.json")

    if not os.path.exists(art_path):
        return None, None

    engine = joblib.load(art_path)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return engine, metrics


engine, metrics = load_engine()

# Header
st.markdown('<div class="main-header">⚡ DynaPrice AI — Dynamic Pricing & Demand Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Log-Log Price Elasticity Modeling • Gradient Boosting Demand Forecasting • Real-Time Profit Maximization Simulator</div>', unsafe_allow_html=True)

if engine is None:
    st.error("Model artifacts not found. Please run the training pipeline first: `python -m src.train`")
    st.stop()

optimizer = engine["optimizer"]
forecaster = engine["forecaster"]
elasticity_analyzer = engine["elasticity_analyzer"]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Interactive Pricing Simulator",
    "📊 Catalog Batch Repricer",
    "📈 Econometric Elasticity Matrix",
    "⚙️ Model Diagnostics & Architecture"
])

# -------------------------------------------------------------
# TAB 1: INTERACTIVE SIMULATOR
# -------------------------------------------------------------
with tab1:
    st.markdown("### 🎛️ Product Pricing Optimization Simulator")
    st.caption("Adjust cost structure, market conditions, and competitor signals to observe real-time profit equilibrium shifts.")

    col_ctrl, col_viz = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("Product Parameters")
        category = st.selectbox(
            "Product Category",
            options=["Consumer Electronics", "Luxury Apparel", "FMCG / Groceries", "Office Supplies", "Home & Living"],
            index=0
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            unit_cost = st.number_input("Unit Cost ($)", min_value=1.0, max_value=2000.0, value=75.0, step=5.0)
            list_price = st.number_input("MSRP / List Price ($)", min_value=1.0, max_value=3000.0, value=149.99, step=5.0)
        with col_c2:
            current_price = st.number_input("Current Offered Price ($)", min_value=1.0, max_value=3000.0, value=139.99, step=5.0)
            competitor_price = st.number_input("Competitor Price ($)", min_value=1.0, max_value=3000.0, value=134.99, step=5.0)

        st.subheader("Market Dynamics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            inventory = st.slider("Stock Inventory", min_value=10, max_value=1000, value=250, step=10)
            customer_rating = st.slider("Rating (1-5★)", min_value=1.0, max_value=5.0, value=4.4, step=0.1)
        with col_m2:
            ad_spend = st.slider("Daily Ad Spend ($)", min_value=0.0, max_value=300.0, value=35.0, step=5.0)
            is_promo = st.checkbox("Holiday / Promo Campaign", value=False)
            is_weekend = st.checkbox("Weekend Traffic Surge", value=False)

        st.subheader("Optimization Goal")
        objective = st.radio("Primary Business Objective", ["profit", "revenue"], horizontal=True)

        opt_result = optimizer.optimize_price(
            category=category,
            unit_cost=unit_cost,
            list_price=list_price,
            current_offered_price=current_price,
            competitor_price=competitor_price,
            inventory_level=inventory,
            customer_rating=customer_rating,
            is_weekend=int(is_weekend),
            is_holiday_promo=int(is_promo),
            ad_spend=ad_spend,
            optimization_objective=objective,
        )

    with col_viz:
        # Top KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                label="Recommended Price",
                value=f"${opt_result['optimal_price']:.2f}",
                delta=f"${opt_result['optimal_price'] - current_price:+.2f} ({(opt_result['optimal_price'] - current_price)/current_price*100:+.1f}%)"
            )
        with kpi2:
            st.metric(
                label="Projected Gross Profit",
                value=f"${opt_result['optimal_projected_profit']:,.2f}",
                delta=f"{opt_result['profit_uplift_pct']:+.1f}% Uplift",
                delta_color="normal" if opt_result['profit_uplift_pct'] >= 0 else "inverse"
            )
        with kpi3:
            st.metric(
                label="Expected Daily Demand",
                value=f"{opt_result['optimal_expected_units']:.1f} units",
                delta=f"{opt_result['optimal_expected_units'] - opt_result['current_expected_units']:+.1f} units"
            )
        with kpi4:
            st.metric(
                label="Gross Margin %",
                value=f"{opt_result['optimal_margin_pct']:.1f}%",
                delta=f"{opt_result['optimal_margin_pct'] - opt_result['current_margin_pct']:+.1f}%"
            )

        # Recommendation Banner
        action = opt_result["strategy_action"]
        if action == "MARKUP":
            badge_class = "rec-badge-markup"
            icon = "📈"
        elif action == "MARKDOWN":
            badge_class = "rec-badge-markdown"
            icon = "🏷️"
        else:
            badge_class = "rec-badge-maintain"
            icon = "⚖️"

        st.markdown(
            f'<div class="{badge_class}">{icon} STRATEGY: {action}</div> — {opt_result["strategy_reason"]}',
            unsafe_allow_html=True
        )

        st.write("")
        # Curve Plot
        curve = opt_result["simulation_curve"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Demand Subplot
        ax1.plot(curve["prices"], curve["demands"], color="#1976D2", lw=2.5, label="Expected Demand Curve")
        ax1.axvline(opt_result["optimal_price"], color="#D81B60", lw=2, linestyle="--", label=f"Optimal Price (${opt_result['optimal_price']:.2f})")
        ax1.axvline(current_price, color="#78909C", lw=1.5, linestyle=":", label=f"Current Price (${current_price:.2f})")
        ax1.axvline(competitor_price, color="#FFB300", lw=1.5, linestyle="-.", label=f"Competitor (${competitor_price:.2f})")
        ax1.set_ylabel("Daily Units Sold", fontsize=10, fontweight="bold")
        ax1.set_title("Price vs. Predicted Sales Volume (Elasticity Response)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", framealpha=0.9)

        # Profit & Revenue Subplot
        ax2.plot(curve["prices"], curve["profits"], color="#00796B", lw=2.5, label="Gross Profit ($)")
        ax2.plot(curve["prices"], curve["revenues"], color="#FB8C00", lw=1.8, linestyle="--", label="Gross Revenue ($)")
        ax2.axvline(opt_result["optimal_price"], color="#D81B60", lw=2, linestyle="--")
        ax2.scatter([opt_result["optimal_price"]], [opt_result["optimal_projected_profit"]], color="#D81B60", s=80, zorder=5)
        ax2.set_xlabel("Offered Price ($)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Financial Yield ($)", fontsize=10, fontweight="bold")
        ax2.set_title("Financial Curves: Profit Maximization Equilibrium", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# -------------------------------------------------------------
# TAB 2: CATALOG BATCH REPRICER
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📊 Multi-SKU Catalog Automated Repricing")
    st.caption("Demonstrates portfolio-wide automated repricing with business guardrails and margin protection.")

    catalog_data = [
        {"Product": "Ultra-HD 4K Monitor 27-inch", "Category": "Consumer Electronics", "Cost": 160.0, "MSRP": 299.99, "Current Price": 279.99, "Competitor Price": 269.99, "Stock": 140, "Rating": 4.6},
        {"Product": "ANC Wireless Headphones", "Category": "Consumer Electronics", "Cost": 55.0, "MSRP": 119.99, "Current Price": 115.00, "Competitor Price": 125.00, "Stock": 380, "Rating": 4.5},
        {"Product": "Italian Cashmere Trench Coat", "Category": "Luxury Apparel", "Cost": 120.0, "MSRP": 420.00, "Current Price": 380.00, "Competitor Price": 395.00, "Stock": 35, "Rating": 4.8},
        {"Product": "Handcrafted Leather Chelsea Boots", "Category": "Luxury Apparel", "Cost": 65.0, "MSRP": 220.00, "Current Price": 195.00, "Competitor Price": 210.00, "Stock": 60, "Rating": 4.7},
        {"Product": "Organic Single-Origin Coffee 1kg", "Category": "FMCG / Groceries", "Cost": 9.5, "MSRP": 22.00, "Current Price": 20.00, "Competitor Price": 18.50, "Stock": 450, "Rating": 4.4},
        {"Product": "Artisan Almond Butter 500g", "Category": "FMCG / Groceries", "Cost": 4.2, "MSRP": 11.50, "Current Price": 10.99, "Competitor Price": 9.99, "Stock": 500, "Rating": 4.3},
        {"Product": "Ergonomic Mesh Executive Chair", "Category": "Office Supplies", "Cost": 80.0, "MSRP": 189.99, "Current Price": 169.99, "Competitor Price": 159.99, "Stock": 90, "Rating": 4.5},
        {"Product": "Cast Iron Dutch Oven 6-Qt", "Category": "Home & Living", "Cost": 35.0, "MSRP": 89.99, "Current Price": 79.99, "Competitor Price": 74.99, "Stock": 120, "Rating": 4.7},
    ]

    reprice_records = []
    total_curr_profit = 0.0
    total_opt_profit = 0.0

    for item in catalog_data:
        res = optimizer.optimize_price(
            category=item["Category"],
            unit_cost=item["Cost"],
            list_price=item["MSRP"],
            current_offered_price=item["Current Price"],
            competitor_price=item["Competitor Price"],
            inventory_level=item["Stock"],
            customer_rating=item["Rating"],
        )
        total_curr_profit += res["current_projected_profit"]
        total_opt_profit += res["optimal_projected_profit"]

        reprice_records.append({
            "Product": item["Product"],
            "Category": item["Category"],
            "Unit Cost": f"${item['Cost']:.2f}",
            "Current Price": f"${item['Current Price']:.2f}",
            "Competitor": f"${item['Competitor Price']:.2f}",
            "ML Optimal Price": f"${res['optimal_price']:.2f}",
            "Price Delta": f"${res['optimal_price'] - item['Current Price']:+.2f}",
            "Profit Uplift": f"{res['profit_uplift_pct']:+.1f}%",
            "Action": res["strategy_action"],
            "Rationale": res["strategy_reason"]
        })

    cat_df = pd.DataFrame(reprice_records)

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Total Catalog Daily Current Profit", f"${total_curr_profit:,.2f}")
    with mcol2:
        st.metric("Total Catalog Daily Optimized Profit", f"${total_opt_profit:,.2f}")
    with mcol3:
        overall_gain = ((total_opt_profit - total_curr_profit) / total_curr_profit * 100) if total_curr_profit > 0 else 0
        st.metric("Portfolio Gross Profit Uplift", f"{overall_gain:+.2f}%", delta=f"${total_opt_profit - total_curr_profit:+,.2f} / day")

    st.dataframe(cat_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: ECONOMETRIC ELASTICITY MATRIX
# -------------------------------------------------------------
with tab3:
    st.markdown("### 📈 Category Price Elasticity of Demand (PED)")
    st.markdown(r"""
    The engine estimates elasticity coefficients using log-log ordinary least squares regression:
    $$\ln(Q) = \alpha + \beta_{\text{PED}} \cdot \ln(P) + \gamma_{\text{cross}} \cdot \ln(P_{\text{competitor}})$$
    """)

    cat_metrics = elasticity_analyzer.category_models
    cards = []
    for cat, data in cat_metrics.items():
        cards.append({
            "Category": cat,
            "Price Elasticity (PED)": round(data["ped"], 2),
            "Cross-Price Elasticity": round(data["cross_ped"], 2),
            "Regression Fit (R²)": round(data["r2"], 3),
            "Economic Sensitivity Profile": data["elasticity_class"],
        })

    ped_df = pd.DataFrame(cards)
    st.table(ped_df)

    st.info("""
    💡 **Economic Interpretation:**
    - **Luxury Apparel (PED ~ -0.74):** Inelastic. Price increases produce minimal demand degradation, enabling aggressive margin expansion.
    - **Consumer Electronics (PED ~ -2.39):** Highly Elastic. Strong consumer cross-shopping; price sensitivity is extreme, meaning micro-discounts yield major volume surges.
    - **Cross-Price Elasticity > 0:** Confirms competitor products act as direct economic substitutes.
    """)

# -------------------------------------------------------------
# TAB 4: MODEL DIAGNOSTICS & ARCHITECTURE
# -------------------------------------------------------------
with tab4:
    st.markdown("### ⚙️ Production Machine Learning Architecture")

    if metrics:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Model Algorithm", "GradientBoostingRegressor")
        with d2:
            st.metric("Test R² Score", f"{metrics.get('r2_score', 0):.4f}")
        with d3:
            st.metric("Test MAE", f"{metrics.get('mae', 0):.2f} units")
        with d4:
            st.metric("Training Samples", f"{metrics.get('train_samples', 0):,} rows")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    diag_img_path = os.path.join(base_dir, "artifacts", "model_diagnostics.png")
    if os.path.exists(diag_img_path):
        st.image(diag_img_path, caption="Calibration Scatter Plot & Relative Feature Importances", use_column_width=True)

    st.markdown("#### FastAPI Microservice Integration Snippet")
    st.code("""
import requests

payload = {
    "category": "Consumer Electronics",
    "unit_cost": 85.0,
    "list_price": 149.99,
    "current_offered_price": 139.99,
    "competitor_price": 145.00,
    "inventory_level": 250,
    "customer_rating": 4.4,
    "optimization_objective": "profit"
}

response = requests.post("http://localhost:8000/optimize-price", json=payload)
print(response.json())
    """, language="python")

st.markdown("---")
st.caption("⚡ Built with Scikit-Learn, FastAPI & Streamlit | DynaPrice AI Portfolio Series")
