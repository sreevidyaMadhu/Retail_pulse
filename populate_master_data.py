import os
import pandas as pd
from werkzeug.security import generate_password_hash

from app import app
from models import db, Shop, User, Product


# --------------------------------------------------
# Product master file
# --------------------------------------------------

PRODUCT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ml",
    "product_master.csv"
)


# --------------------------------------------------
# Main database population
# --------------------------------------------------

with app.app_context():

    print("\n==============================================")
    print("RetailPulse - Master Data Setup")
    print("==============================================\n")

    # --------------------------------------------------
    # 1. Check product master file
    # --------------------------------------------------

    print("Loading product master...")

    if not os.path.exists(PRODUCT_FILE):
        print("ERROR: product_master.csv not found!")
        print("Expected location:")
        print(PRODUCT_FILE)
        exit()

    df = pd.read_csv(PRODUCT_FILE)

    print("Products found:", len(df))

    # --------------------------------------------------
    # 2. Create Shop
    # --------------------------------------------------

    shop = Shop(
        shop_name="City Central Supermarket",
        city="Thiruvananthapuram",
        latitude=8.5241,
        longitude=76.9366
    )

    db.session.add(shop)
    db.session.flush()

    print("Shop created:", shop.shop_name)

    # --------------------------------------------------
    # 3. Create Manager
    # --------------------------------------------------

    manager = User(
        shop_id=shop.id,
        username="manager1",
        password_hash=generate_password_hash("manager123"),
        role="manager"
    )

    # --------------------------------------------------
    # 4. Create Employee
    # --------------------------------------------------

    employee = User(
        shop_id=shop.id,
        username="employee1",
        password_hash=generate_password_hash("employee123"),
        role="employee"
    )

    db.session.add(manager)
    db.session.add(employee)

    print("Manager created: manager1")
    print("Employee created: employee1")

    # --------------------------------------------------
    # 5. Create Products
    # --------------------------------------------------

    product_count = 0

    for _, row in df.iterrows():

        product = Product(
            shop_id=shop.id,
            name=row["product_name"],
            category=row["category"],
            price=0,
            weather_dependency=row["weather_dependency"]
        )

        db.session.add(product)
        product_count += 1

    # --------------------------------------------------
    # 6. Save everything
    # --------------------------------------------------

    db.session.commit()

    # --------------------------------------------------
    # 7. Success message
    # --------------------------------------------------

    print("\n==============================================")
    print("Master data created successfully!")
    print("==============================================")

    print("Shop       :", shop.shop_name)
    print("City       :", shop.city)
    print("Manager    : manager1")
    print("Employee   : employee1")
    print("Products   :", product_count)

    print("\nLogin details")
    print("----------------------------------------------")
    print("Manager")
    print("Username : manager1")
    print("Password : manager123")

    print("\nEmployee")
    print("Username : employee1")
    print("Password : employee123")

    print("\n==============================================")