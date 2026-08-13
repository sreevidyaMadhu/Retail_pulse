from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from datetime import datetime
from models import Sale, db, Shop, Product, User, Inventory, InventoryBatch
from weather_service import fetch_weather_by_city
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

    return render_template("manager_dashboard.html")
@app.route("/employee/dashboard")
@login_required
def employee_dashboard():

    if current_user.role != "employee":
        return "Access Denied", 403

    return render_template("employee_dashboard.html")

@app.route("/dashboard")
@login_required
def dashboard():
    # 1. Fetch live weather for logged-in shop's city
    
    # 2. Get stock for logged-in shop only
    weather = fetch_weather_by_city(current_user.shop.city)

    products = Product.query.filter_by(shop_id=current_user.shop_id).all()
    return render_template("dashboard.html", shop=current_user, weather=weather, products=products)
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
        reorder_level = int(request.form["reorder_level"])

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
            product_id=product.id,
            quantity=quantity,
            reorder_level=reorder_level
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
    ).all()

    return render_template(
        "sales.html",
        sales=sales
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

        product = Product.query.filter_by(
            id=product_id,
            shop_id=current_user.shop_id
        ).first_or_404()

        inventory = Inventory.query.filter_by(
            product_id=product.id
        ).first_or_404()

        if quantity > inventory.quantity:
            return "Not enough stock available", 400

        # Get available batches ordered by earliest expiry
        batches = InventoryBatch.query.filter(
            InventoryBatch.inventory_id == inventory.id,
            InventoryBatch.quantity > 0,
            InventoryBatch.expiry_date >= datetime.utcnow().date()
        ).order_by(
            InventoryBatch.expiry_date.asc()
        ).all()

        remaining = quantity

        for batch in batches:

            if remaining == 0:
                break

            amount_from_batch = min(
                remaining,
                batch.quantity
            )

            batch.quantity -= amount_from_batch
            remaining -= amount_from_batch

        # Safety check
        if remaining > 0:
            db.session.rollback()
            return "Not enough usable batch stock available", 400

        # Reduce total inventory
        inventory.quantity -= quantity

        # Record the sale
        sale = Sale(
            product_id=product.id,
            batch_id=batches[0].id,
            quantity=quantity,
            sale_price=product.price
        )

        db.session.add(sale)

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