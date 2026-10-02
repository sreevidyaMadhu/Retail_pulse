from datetime import date

from app import app, get_sales_features


with app.app_context():

    product_id = 1

    prediction_date = date(2026, 2, 8)

    features = get_sales_features(
        product_id,
        prediction_date
    )

    print("=" * 50)
    print("Sales Feature Test")
    print("=" * 50)

    print("Product ID:", product_id)
    print("Prediction Date:", prediction_date)

    print(
        "Previous Day Sales:",
        features["previous_day_sales"]
    )

    print(
        "Rolling 7-Day Sales:",
        features["rolling_7day_sales"]
    )

    print("=" * 50)