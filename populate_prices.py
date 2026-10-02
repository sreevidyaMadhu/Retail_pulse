from app import app
from models import db, Product


# --------------------------------------------------
# Product prices
# These are assumed supermarket selling prices
# for project/demo purposes.
# --------------------------------------------------

PRODUCT_PRICES = {

    # Beverages
    "Coca Cola": 45,
    "Pepsi": 45,
    "Sprite": 45,
    "Fruit Juice": 60,
    "Packaged Drinking Water": 20,
    "Energy Drink": 120,
    "Lemon Juice": 40,
    "Flavoured Milk": 35,

    # Ice Cream & Frozen
    "Vanilla Ice Cream": 180,
    "Chocolate Ice Cream": 200,
    "Strawberry Ice Cream": 190,
    "Kulfi": 50,
    "Ice Cream Cone": 40,
    "Frozen Dessert": 150,

    # Fruits
    "Watermelon": 50,
    "Mango": 100,
    "Orange": 90,
    "Grapes": 120,
    "Pineapple": 80,
    "Banana": 60,
    "Papaya": 70,
    "Apple": 180,

    # Vegetables
    "Tomato": 50,
    "Cucumber": 50,
    "Carrot": 60,
    "Potato": 40,
    "Onion": 45,
    "Green Chilli": 80,
    "Spinach": 30,
    "Capsicum": 90,

    # Dairy
    "Milk": 30,
    "Curd": 40,
    "Butter": 60,
    "Cheese": 120,
    "Paneer": 100,
    "Buttermilk": 25,
    "Flavoured Yogurt": 50,
    "Fresh Cream": 90,

    # Bakery
    "Bread": 50,
    "Bun": 30,
    "Burger Bun": 45,
    "Cake": 300,
    "Rusk": 60,
    "Biscuits": 30,
    "Croissant": 80,
    "Bread Rolls": 50,

    # Snacks
    "Potato Chips": 30,
    "Mixture": 80,
    "Namkeen": 70,
    "Popcorn": 40,
    "Chocolate": 80,
    "Candy": 5,
    "Nuts": 180,
    "Instant Snack Mix": 70,
    "Instant Noodles": 20,
    "Soup Mix": 60,
    "Pasta": 80,
    "Ready-to-Eat Meals": 120,
    "Canned Food": 100,
    "Breakfast Cereal": 250,

    # Rain / seasonal products
    "Umbrella": 250,
    "Raincoat": 600,
    "Waterproof Shoe Cover": 150,
    "Candles": 30,
    "Batteries": 50,
    "Mosquito Repellent": 150,

    # Personal care / household
    "Soap": 40,
    "Shampoo": 120,
    "Conditioner": 140,
    "Toothpaste": 100,
    "Detergent": 120,
    "Dishwashing Liquid": 100,
    "Floor Cleaner": 150,

    # Grocery
    "Rice": 70,
    "Wheat Flour": 60,
    "Sugar": 50,
    "Salt": 25,
    "Cooking Oil": 150,
    "Atta": 60,
    "Spices": 80
}


# --------------------------------------------------
# Update database
# --------------------------------------------------

with app.app_context():

    print("\n==============================================")
    print("RetailPulse - Product Price Setup")
    print("==============================================\n")

    products = Product.query.order_by(Product.id).all()

    if not products:
        print("ERROR: No products found.")
        print("Please run populate_master_data.py first.")
        exit()

    print("Products found:", len(products))

    updated = 0
    missing = []

    for product in products:

        if product.name in PRODUCT_PRICES:

            product.price = PRODUCT_PRICES[product.name]

            print(
                f"Updated: {product.name:<30} "
                f"₹{product.price}"
            )

            updated += 1

        else:

            missing.append(product.name)

    # --------------------------------------------------
    # Save changes
    # --------------------------------------------------

    db.session.commit()

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    print("\n==============================================")
    print("Price setup completed!")
    print("==============================================")

    print("Products found :", len(products))
    print("Prices updated :", updated)
    print("Missing prices :", len(missing))

    if missing:

        print("\nProducts without prices:")
        for name in missing:
            print("-", name)

    print("\n==============================================")