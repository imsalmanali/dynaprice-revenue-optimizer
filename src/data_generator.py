"""
Synthetic Retail & E-Commerce Pricing Dataset Generator.
Simulates realistic economic demand functions with category-specific price elasticity,
competitor price sensitivity, promotional lifts, and inventory constraints.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


CATEGORIES_CONFIG: Dict[str, Dict[str, Any]] = {
    "Consumer Electronics": {
        "products": ["Noise-Cancelling Headphones", "4K Ultra-HD Monitor", "Wireless Mechanical Keyboard", "Smart Home Hub", "Portable Bluetooth Speaker"],
        "cost_range": (40.0, 350.0),
        "markup_range": (1.35, 1.80),
        "base_elasticity": -2.25,  # Highly elastic
        "competitor_sensitivity": 1.45,
        "base_daily_demand": 45,
    },
    "Luxury Apparel": {
        "products": ["Cashmere Trench Coat", "Italian Leather Chelsea Boots", "Silk Evening Scarf", "Designer Chronograph Watch", "Tailored Wool Blazer"],
        "cost_range": (50.0, 300.0),
        "markup_range": (2.20, 3.80),
        "base_elasticity": -0.65,  # Inelastic (prestige goods)
        "competitor_sensitivity": 0.40,
        "base_daily_demand": 15,
    },
    "FMCG / Groceries": {
        "products": ["Organic Colombian Roast Coffee", "Extra Virgin Olive Oil 1L", "Artisan Almond Butter", "Hydro-Boost Electrolyte Pack", "Single-Origin Dark Chocolate"],
        "cost_range": (4.0, 22.0),
        "markup_range": (1.25, 1.55),
        "base_elasticity": -1.40,  # Moderate elasticity
        "competitor_sensitivity": 1.15,
        "base_daily_demand": 120,
    },
    "Office Supplies": {
        "products": ["Ergonomic Mesh Task Chair", "Standing Desk Converter", "Dual Monitor Aluminum Arm", "Executive Leather Notebook", "High-Speed Document Scanner"],
        "cost_range": (15.0, 160.0),
        "markup_range": (1.40, 2.00),
        "base_elasticity": -1.10,  # Unitary/Moderate
        "competitor_sensitivity": 0.90,
        "base_daily_demand": 35,
    },
    "Home & Living": {
        "products": ["Sonic Air Purifier", "Cast Iron Dutch Oven 6-Qt", "Percale Cotton Sheet Set", "Stainless Steel Knife Block", "Smart LED Ambient Light Bar"],
        "cost_range": (20.0, 140.0),
        "markup_range": (1.45, 2.10),
        "base_elasticity": -1.65,  # Elastic
        "competitor_sensitivity": 1.10,
        "base_daily_demand": 30,
    },
}


def generate_pricing_dataset(
    n_samples: int = 12000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transactions with realistic microeconomic demand mechanics:
    log(Q) = log(Q0) + beta * log(P / P_msrp) + gamma * log(P_competitor / P) + delta * Features + epsilon
    """
    np.random.seed(random_state)
    records: List[Dict[str, Any]] = []

    categories = list(CATEGORIES_CONFIG.keys())
    cat_proportions = [0.26, 0.16, 0.28, 0.14, 0.16]

    for i in range(n_samples):
        cat = np.random.choice(categories, p=cat_proportions)
        cfg = CATEGORIES_CONFIG[cat]
        prod_name = np.random.choice(cfg["products"])

        unit_cost = float(np.round(np.random.uniform(*cfg["cost_range"]), 2))
        msrp_markup = np.random.uniform(*cfg["markup_range"])
        list_price = float(np.round(unit_cost * msrp_markup, 2))

        # Dynamic pricing strategy variance around list price (-25% to +18%)
        pricing_factor = np.random.uniform(0.75, 1.18)
        offered_price = float(np.round(list_price * pricing_factor, 2))
        # Hard floor: Offered price shouldn't go strictly below cost * 1.05 in standard ops
        offered_price = max(offered_price, np.round(unit_cost * 1.05, 2))

        # Competitor pricing reacts around MSRP (-20% to +20%)
        competitor_factor = np.random.uniform(0.80, 1.20)
        competitor_price = float(np.round(list_price * competitor_factor, 2))

        # Contextual features
        discount_pct = float(np.round(max(0.0, (list_price - offered_price) / list_price * 100), 1))
        inventory_level = int(np.random.randint(5, 500))
        customer_rating = float(np.round(np.random.uniform(3.4, 4.95), 2))
        is_weekend = int(np.random.binomial(1, 2 / 7))
        is_holiday_promo = int(np.random.binomial(1, 0.12))
        ad_spend = float(np.round(np.random.exponential(scale=65.0) if np.random.rand() > 0.3 else 0.0, 2))
        day_of_week = int(np.random.randint(0, 7))

        # Microeconomic log-demand simulation
        price_ratio = offered_price / list_price
        comp_ratio = competitor_price / offered_price

        # Elasticity effect: P_ratio ^ elasticity
        elasticity = cfg["base_elasticity"]
        price_effect = (price_ratio) ** elasticity

        # Competitor cross-elasticity effect: higher competitor price pushes demand to us
        comp_sensitivity = cfg["competitor_sensitivity"]
        comp_effect = (comp_ratio) ** comp_sensitivity

        # Organic & promotional multipliers
        rating_mult = (customer_rating / 4.0) ** 1.3
        weekend_mult = 1.22 if is_weekend else 1.0
        promo_mult = 1.48 if is_holiday_promo else 1.0
        ad_mult = 1.0 + (np.log1p(ad_spend) * 0.055)

        base_q = cfg["base_daily_demand"]
        noise = np.random.lognormal(mean=0.0, sigma=0.18)

        expected_units = base_q * price_effect * comp_effect * rating_mult * weekend_mult * promo_mult * ad_mult * noise
        units_sold = int(np.round(expected_units))
        # Ensure non-negative and bound by physical inventory
        units_sold = min(max(0, units_sold), inventory_level)

        revenue = float(np.round(units_sold * offered_price, 2))
        cogs = float(np.round(units_sold * unit_cost, 2))
        gross_profit = float(np.round(revenue - cogs, 2))
        profit_margin_pct = float(np.round((gross_profit / revenue * 100) if revenue > 0 else 0.0, 2))

        records.append({
            "transaction_id": f"TX-{100000 + i}",
            "category": cat,
            "product_name": prod_name,
            "unit_cost": unit_cost,
            "list_price": list_price,
            "offered_price": offered_price,
            "competitor_price": competitor_price,
            "price_diff_vs_competitor": float(np.round(offered_price - competitor_price, 2)),
            "price_ratio_vs_competitor": float(np.round(offered_price / competitor_price, 4)),
            "discount_pct": discount_pct,
            "inventory_level": inventory_level,
            "customer_rating": customer_rating,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday_promo": is_holiday_promo,
            "ad_spend": ad_spend,
            "units_sold": units_sold,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "profit_margin_pct": profit_margin_pct,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_pricing_dataset(12000)
    print(f"Generated dataset shape: {df.shape}")
    print(df.head(3))
