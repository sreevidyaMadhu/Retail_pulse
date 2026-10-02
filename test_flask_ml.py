from datetime import date

from app import (
    app,
    get_product_demand_prediction
)

from models import Product
from weather_service import fetch_weather_by_city


with app.app_context():

    product = Product.query.get(1)

    weather = fetch_weather_by_city(
        "Thiruvananthapuram"
    )

    prediction = get_product_demand_prediction(
        product,
        date(2026, 2, 8),
        weather
    )

    print("=" * 50)
    print("Flask + ML Integration Test")
    print("=" * 50)

    print("Product:", product.name)
    print("Predicted Demand:", prediction)

    print("=" * 50)