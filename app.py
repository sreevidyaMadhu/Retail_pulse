from flask import Flask, render_template, request, redirect, url_for
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from models import (
    Sale,
    db,
    Shop,
    Product,
    User,
    Inventory,
    InventoryBatch
)

from weather_service import fetch_weather_by_city
from ml2.predict_demand import predict_demand
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    try:
        db.session.execute(db.text("SELECT 1"))
        print("✅ PostgreSQL connection successful!")
    except Exception as e:
        print("❌ PostgreSQL connection failed!")
        print(e)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
def get_sales_features(product_id, prediction_date):
    """
    Calculate recent sales features required by
    the RetailPulse demand prediction model.
    """

    prediction_date = prediction_date.date() if isinstance(
        prediction_date,
        datetime
    ) else prediction_date

    # Previous day's sales
    previous_day = prediction_date - timedelta(days=1)

    previous_day_sales = db.session.query(
        db.func.coalesce(
            db.func.sum(Sale.quantity),
            0
        )
    ).filter(
        Sale.product_id == product_id,
        db.func.date(Sale.sale_date) == previous_day
    ).scalar()

    # Previous 7 days' sales
    seven_days_start = prediction_date - timedelta(days=7)

    seven_day_sales = db.session.query(
        db.func.coalesce(
            db.func.sum(Sale.quantity),
            0
        )
    ).filter(
        Sale.product_id == product_id,
        db.func.date(Sale.sale_date) >= seven_days_start,
        db.func.date(Sale.sale_date) < prediction_date
    ).scalar()

    rolling_7day_sales = seven_day_sales / 7

    return {
        "previous_day_sales": float(previous_day_sales),
        "rolling_7day_sales": float(rolling_7day_sales)
    }


def get_product_demand_prediction(product, prediction_date, weather):
    """
    Generate demand prediction for a product
    using real sales history and weather data.
    """

    sales_features = get_sales_features(
        product.id,
        prediction_date
    )

    predicted_demand = predict_demand(

        product_id=f"P{product.id:03d}",

        category=product.category,

        weather_dependency=product.weather_dependency,

        date=prediction_date,

        previous_day_sales=sales_features[
            "previous_day_sales"
        ],

        rolling_7day_sales=sales_features[
            "rolling_7day_sales"
        ],

        temperature=weather["temp"],

        min_temperature=weather["min_temperature"],

        max_temperature=weather["max_temperature"],

        humidity=weather["humidity"],

        rainfall=weather["rainfall"],

        wind_speed=weather["wind_speed"],

        pressure=weather["pressure"]
    )

    return predicted_demand

def analyze_inventory_risk(product, predicted_demand):
    """
    Compare predicted demand with usable inventory
    and generate an inventory risk recommendation.
    """

    inventory = Inventory.query.filter_by(
        product_id=product.id
    ).first()

    if not inventory:
        return {
            "current_stock": 0,
            "usable_stock": 0,
            "expired_stock": 0,
            "shortage": round(predicted_demand, 2),
            "risk": "Out of Stock",
            "recommendation": "Replenishment Required"
        }

    today = datetime.utcnow().date()

    usable_stock = 0
    expired_stock = 0

    batches = InventoryBatch.query.filter_by(
        inventory_id=inventory.id
    ).all()

    for batch in batches:

        if batch.quantity <= 0:
            continue

        if batch.expiry_date < today:
            expired_stock += batch.quantity
        else:
            usable_stock += batch.quantity

    shortage = max(
        0,
        predicted_demand - usable_stock
    )

    if usable_stock == 0:

        risk = "Out of Stock"
        recommendation = "Replenishment Required"

    elif usable_stock < predicted_demand:

        risk = "Low Stock Risk"
        recommendation = (
            f"Consider replenishing approximately "
            f"{round(shortage)} units"
        )

    else:

        risk = "Stock Sufficient"
        recommendation = "No Immediate Replenishment Required"

    return {
        "current_stock": inventory.quantity,
        "usable_stock": usable_stock,
        "expired_stock": expired_stock,
        "shortage": round(shortage, 2),
        "risk": risk,
        "recommendation": recommendation
    }
    
       
def seed_database():
    """Create tables and insert dummy shop, users, products and inventory."""

    with app.app_context():
        # db.drop_all()

        db.create_all()

        # Only seed if there are no shops
        if not Shop.query.first():

            # -------------------------
            # SHOP
            # -------------------------
            shop1 = Shop(
                shop_name="City Central Supermarket",
                city="Thiruvananthapuram",
                latitude=8.5241,
                longitude=76.9366
            )

            db.session.add(shop1)
            db.session.commit()

            # -------------------------
            # USERS
            # -------------------------
            manager = User(
                shop_id=shop1.id,
                username="manager1",
                password_hash=generate_password_hash("manager123"),
                role="manager"
            )

            employee = User(
                shop_id=shop1.id,
                username="employee1",
                password_hash=generate_password_hash("employee123"),
                role="employee"
            )

            db.session.add_all([manager, employee])
            db.session.commit()

            # -------------------------
            # PRODUCTS
            # -------------------------
            p1 = Product(
                shop_id=shop1.id,
                name="Bottled Water 1L",
                category="Beverage",
                price=20,
                weather_dependency="Hot Weather"
            )

            p2 = Product(
                shop_id=shop1.id,
                name="Compact Umbrella",
                category="Rain Accessories",
                price=250,
                weather_dependency="Rainy Weather"
            )

            db.session.add_all([p1, p2])
            db.session.commit()

            # -------------------------
            # INVENTORY
            # -------------------------
            inventory1 = Inventory(
                product_id=p1.id,
                quantity=15,
                reorder_level=10
            )

            inventory2 = Inventory(
                product_id=p2.id,
                quantity=8,
                reorder_level=10
            )

            db.session.add_all([inventory1, inventory2])
            db.session.commit()

            print("✅ Database seeded successfully!")

# --- ROUTES ---

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):

            login_user(user)

            if user.role == "manager":
                return redirect(url_for("manager_dashboard"))

            elif user.role == "employee":
                return redirect(url_for("employee_dashboard"))

        return "Invalid username or password"

    return render_template("login.html")
@app.route("/manager/dashboard")
@login_required
def manager_dashboard():

    if current_user.role != "manager":
        return "Access Denied", 403

    shop = current_user.shop

    # --------------------------------------------------
    # Live weather
    # --------------------------------------------------

    weather = fetch_weather_by_city(
        shop.city
    )

   # --------------------------------------------------
# Products
# --------------------------------------------------

    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    for product in products:
        product.current_stock = (
            product.inventory.quantity
            if product.inventory
            else 0
        )


    # --------------------------------------------------
    # AI demand + inventory analysis
    # --------------------------------------------------

    prediction_date = datetime.utcnow().date()

    ai_products = []

    if weather:

        for product in products:

            try:

                predicted_demand = (
                    get_product_demand_prediction(
                        product,
                        prediction_date,
                        weather
                    )
                )

                risk = analyze_inventory_risk(
                    product,
                    predicted_demand
                )

                ai_products.append({
                    "product": product,
                    "predicted_demand": predicted_demand,
                    "current_stock": risk[
                        "current_stock"
                    ],
                    "usable_stock": risk[
                        "usable_stock"
                    ],
                    "expired_stock": risk[
                        "expired_stock"
                    ],
                    "shortage": risk[
                        "shortage"
                    ],
                    "risk": risk[
                        "risk"
                    ],
                    "recommendation": risk[
                        "recommendation"
                    ]
                })

            except Exception as e:

                print(
                    f"Prediction failed for "
                    f"{product.name}: {e}"
                )
    ai_products.sort(
    key=lambda item: item["shortage"],
    reverse=True
)

    return render_template(
        "manager_dashboard.html",
        shop=shop,
        weather=weather,
        products=products,
        ai_products=ai_products
    )
@app.route("/employee/dashboard")
@login_required
def employee_dashboard():

    if current_user.role != "employee":
        return "Access Denied", 403

    shop = current_user.shop

    weather = fetch_weather_by_city(shop.city)

    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    for product in products:
        product.current_stock = (
            product.inventory.quantity
            if product.inventory
            else 0
        )

    return render_template(
        "employee_dashboard.html",
        shop=shop,
        weather=weather,
        products=products
    )

@app.route("/dashboard")
@login_required
def dashboard():
    # 1. Fetch live weather for logged-in shop's city
    
    # 2. Get stock for logged-in shop only
    weather = fetch_weather_by_city(current_user.shop.city)

    products = Product.query.filter_by(shop_id=current_user.shop_id).all()
    return render_template("dashboard.html", shop=current_user, weather=weather, products=products)
@app.route("/manager/weather")
@login_required
def weather_insights():

    if current_user.role != "manager":
        return "Access Denied", 403

    # Logged-in manager's shop
    shop = current_user.shop

    # Get live weather
    weather = fetch_weather_by_city(shop.city)

    # Get products belonging to this shop
    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    # Only products having weather dependency
    weather_products = [
        product
        for product in products
        if product.weather_dependency
    ]

    return render_template(
        "weather_insights.html",
        shop=shop,
        weather=weather,
        products=weather_products
    )
@app.route("/manager/demand")
@login_required
def demand_prediction():

    if current_user.role != "manager":
        return "Access Denied", 403

    shop = current_user.shop

    # Get live weather
    weather = fetch_weather_by_city(shop.city)

    # Get products for this shop
    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    prediction_date = datetime.utcnow().date()

    demand_predictions = []

    if weather:

        for product in products:

            try:

                predicted_demand = get_product_demand_prediction(
                    product,
                    prediction_date,
                    weather
                )

                sales_features = get_sales_features(
                    product.id,
                    prediction_date
                )

                demand_predictions.append({
                    "product": product,
                    "predicted_demand": predicted_demand,
                    "previous_day_sales":
                        sales_features["previous_day_sales"],
                    "rolling_7day_sales":
                        sales_features["rolling_7day_sales"]
                })

            except Exception as e:

                print(
                    f"Demand prediction failed for "
                    f"{product.name}: {e}"
                )

    # Highest predicted demand first
    demand_predictions.sort(
        key=lambda item: item["predicted_demand"],
        reverse=True
    )

    return render_template(
        "demand_prediction.html",
        shop=shop,
        weather=weather,
        demand_predictions=demand_predictions
    )
@app.route("/manager/products")
@login_required
def products():

    if current_user.role != "manager":
        return "Access Denied", 403

    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    return render_template(
        "products.html",
        products=products
    )
@app.route("/manager/products/add", methods=["GET", "POST"])
@login_required
def add_product():

    if current_user.role != "manager":
        return "Access Denied", 403

    if request.method == "POST":

        # Product details
        name = request.form["name"]
        category = request.form["category"]
        price = float(request.form["price"])

        # Initial inventory details
        quantity = int(request.form["quantity"])

        # Initial batch details
        batch_number = request.form["batch_number"]
        manufacture_date = request.form["manufacture_date"]
        expiry_date = request.form["expiry_date"]
        received_date = request.form["received_date"]

        # 1. Create Product
        product = Product(
            shop_id=current_user.shop_id,
            name=name,
            category=category,
            price=price
        )

        db.session.add(product)
        db.session.flush()

        # 2. Create Inventory
        inventory = Inventory(
            product_id=product.id
        )

        db.session.add(inventory)
        db.session.flush()

        # 3. Create first batch
        batch = InventoryBatch(
            inventory_id=inventory.id,
            batch_number=batch_number,
            quantity=quantity,
            manufacture_date=manufacture_date,
            expiry_date=expiry_date,
            received_date=received_date
        )

        db.session.add(batch)

        # Save everything
        db.session.commit()

        return redirect(url_for("products"))

    return render_template("add_product.html")
# @app.route("/manager/products/add", methods=["GET", "POST"])
# @login_required
# def add_product():

#     if current_user.role != "manager":
#         return "Access Denied", 403

#     if request.method == "POST":

#         name = request.form["name"]
#         category = request.form["category"]
#         price = request.form["price"]

#         product = Product(
#             shop_id=current_user.shop_id,
#             name=name,
#             category=category,
#             price=float(price)
#         )

#         db.session.add(product)
#         db.session.commit()

#         return redirect(url_for("products"))

#     return render_template("add_product.html")
@app.route("/manager/inventory")
@login_required
def inventory():

    if current_user.role not in ["manager", "employee"]:
        return "Access Denied", 403

    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    inventory_data = []

    for product in products:

        stock = Inventory.query.filter_by(
            product_id=product.id
        ).first()

        if stock:
            inventory_data.append({
                "product": product,
                "stock": stock
            })

    return render_template(
        "inventory.html",
        inventory_data=inventory_data 
    )
@app.route("/manager/inventory/<int:product_id>/batches")
@login_required
def view_batches(product_id):

    if current_user.role not in ["manager", "employee"]:
        return "Access Denied", 403

    product = Product.query.filter_by(
        id=product_id,
        shop_id=current_user.shop_id
    ).first_or_404()

    inventory = Inventory.query.filter_by(
        product_id=product.id
    ).first_or_404()

    batches = InventoryBatch.query.filter_by(
        inventory_id=inventory.id
    ).order_by(
        InventoryBatch.expiry_date
    ).all()

    today = datetime.utcnow().date()

    return render_template(
        "batches.html",
        product=product,
        inventory=inventory,
        batches=batches,
        today=today
    )
@app.route("/manager/inventory/<int:product_id>/add-batch", methods=["GET", "POST"])
@login_required
def add_batch(product_id):

    if current_user.role not in ["manager", "employee"]:
        return "Access Denied", 403

    product = Product.query.filter_by(
        id=product_id,
        shop_id=current_user.shop_id
    ).first_or_404()

    inventory = Inventory.query.filter_by(
        product_id=product.id
    ).first_or_404()

    if request.method == "POST":

        batch_number = request.form["batch_number"]
        quantity = int(request.form["quantity"])
        manufacture_date = request.form["manufacture_date"]
        received_date = request.form["received_date"]
        expiry_date = request.form["expiry_date"]

        batch = InventoryBatch(
            inventory_id=inventory.id,
            batch_number=batch_number,
            quantity=quantity,
            manufacture_date=manufacture_date,
            received_date=received_date,
            expiry_date=expiry_date
        )

        db.session.add(batch)

        # Automatically increase total inventory
        inventory.quantity += quantity

        db.session.commit()

        return redirect(
            url_for(
                "view_batches",
                product_id=product.id
            )
        )

    return render_template(
        "add_batch.html",
        product=product
    )
@app.route("/manager/sales")
@login_required
def sales():

    if current_user.role not in ["manager", "employee"]:
        return "Access Denied", 403

    sales = Sale.query.join(
        Product,
        Sale.product_id == Product.id
    ).filter(
        Product.shop_id == current_user.shop_id
    ).order_by(
        Sale.sale_date.desc()
    ).all()

    total_sales_value = sum(
        sale.quantity * sale.sale_price
        for sale in sales
    )

    return render_template(
        "sales.html",
        sales=sales,
        total_sales_value=total_sales_value
    )
@app.route("/manager/sales/add", methods=["GET", "POST"])
@login_required
def add_sale():
    if current_user.role not in ["manager", "employee"]:
        return "Access Denied", 403

    products = Product.query.filter_by(
        shop_id=current_user.shop_id
    ).all()

    if request.method == "POST":

        product_id = int(request.form["product_id"])
        quantity = int(request.form["quantity"])

        if quantity <= 0:
            return "Quantity must be greater than 0", 400

        # Make sure the product belongs to the logged-in user's shop
        product = Product.query.filter_by(
            id=product_id,
            shop_id=current_user.shop_id
        ).first_or_404()

        inventory = Inventory.query.filter_by(
            product_id=product.id
        ).first_or_404()

        # Check total available stock
        if quantity > inventory.quantity:
            return "Not enough stock available", 400

        # Get usable batches in FEFO order
        batches = InventoryBatch.query.filter(
            InventoryBatch.inventory_id == inventory.id,
            InventoryBatch.quantity > 0,
            InventoryBatch.expiry_date >= datetime.utcnow().date()
        ).order_by(
            InventoryBatch.expiry_date.asc()
        ).all()

        remaining = quantity

        # Store the sale records that will be created
        sale_records = []

        for batch in batches:

            if remaining == 0:
                break

            # Quantity taken from this batch
            amount_from_batch = min(
                remaining,
                batch.quantity
            )

            # Reduce batch stock
            batch.quantity -= amount_from_batch

            # Reduce remaining quantity to sell
            remaining -= amount_from_batch

            # Create a separate sale record for this batch
            sale = Sale(
                product_id=product.id,
                batch_id=batch.id,
                quantity=amount_from_batch,
                sale_price=product.price
            )

            sale_records.append(sale)

        # This protects us if the inventory quantity says enough
        # but usable batches do not actually contain enough stock.
        if remaining > 0:
            db.session.rollback()
            return "Not enough usable batch stock available", 400

        # Update total inventory
        inventory.quantity -= quantity

        # Add all batch-specific sale records
        db.session.add_all(sale_records)

        # Save everything together
        db.session.commit()

        return redirect(url_for("sales"))

    return render_template(
        "add_sale.html",
        products=products
    )

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    seed_database()
    app.run(debug=True)