from datetime import date

from app import (
    app,
    get_product_demand_prediction,
    analyze_inventory_risk
)

from models import Product
from weather_service import fetch_weather_by_city


with app.app_context():

    print("=" * 60)
    print("RetailPulse Inventory Risk Test")
    print("=" * 60)

    product = Product.query.get(1)

    prediction_date = date(2026, 2, 8)

    weather = fetch_weather_by_city(
        "Thiruvananthapuram"
    )

    predicted_demand = get_product_demand_prediction(
        product,
        prediction_date,
        weather
    )

    risk = analyze_inventory_risk(
        product,
        predicted_demand
    )

    print("\nProduct")
    print("-" * 40)
    print("Name:", product.name)

    print("\nDemand Prediction")
    print("-" * 40)
    print(
        "Predicted Demand:",
        predicted_demand,
        "units"
    )

    print("\nInventory")
    print("-" * 40)
    print(
        "Current Stock:",
        risk["current_stock"]
    )

    print(
        "Usable Stock:",
        risk["usable_stock"]
    )

    print(
        "Expired Stock:",
        risk["expired_stock"]
    )

    print("\nRisk Analysis")
    print("-" * 40)
    print(
        "Shortage:",
        risk["shortage"],
        "units"
    )

    print(
        "Risk:",
        risk["risk"]
    )

    print(
        "Recommendation:",
        risk["recommendation"]
    )

    print("=" * 60)