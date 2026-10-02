import random
from datetime import date, timedelta

from app import app
from models import db, Product, Inventory, InventoryBatch


# --------------------------------------------------
# Product categories with approximate expiry ranges
# --------------------------------------------------

def get_expiry_days(category):
    """
    Return a realistic expiry range based on product category.
    """

    category = category.lower()

    # Highly perishable products
    if any(word in category for word in [
        "ice cream",
        "frozen"
    ]):
        return random.randint(60, 180)

    if any(word in category for word in [
        "dairy"
    ]):
        return random.randint(7, 30)

    if any(word in category for word in [
        "bakery"
    ]):
        return random.randint(3, 15)

    # General food products
    if any(word in category for word in [
        "beverages",
        "fruits",
        "vegetables"
    ]):
        return random.randint(5, 30)

    # Household / personal care
    if any(word in category for word in [
        "household",
        "personal care"
    ]):
        return random.randint(180, 730)

    # Default
    return random.randint(180, 730)


# --------------------------------------------------
# Main
# --------------------------------------------------

with app.app_context():

    print("\n==============================================")
    print("RetailPulse - Inventory & Batch Setup")
    print("==============================================\n")

    products = Product.query.order_by(Product.id).all()

    if not products:
        print("ERROR: No products found.")
        print("Please run populate_master_data.py first.")
        exit()

    print("Products found:", len(products))

    total_inventory = 0
    total_batches = 0

    today = date.today()

    # --------------------------------------------------
    # Create inventory and batches
    # --------------------------------------------------

    for product in products:

        # ----------------------------------------------
        # Current stock
        # ----------------------------------------------

        current_stock = random.randint(20, 120)

        inventory = Inventory(
            product_id=product.id,
            quantity=current_stock
        )

        db.session.add(inventory)
        db.session.flush()

        # ----------------------------------------------
        # Split stock into 2 batches
        # ----------------------------------------------

        first_batch_quantity = int(current_stock * 0.60)
        second_batch_quantity = current_stock - first_batch_quantity

        for batch_number, quantity in [
            ("B001", first_batch_quantity),
            ("B002", second_batch_quantity)
        ]:

            # ------------------------------------------
            # Expiry period
            # ------------------------------------------

            expiry_days = get_expiry_days(product.category)

            # ------------------------------------------
            # Manufacture date
            #
            # Keep the product reasonably recent.
            # The maximum age is at most half of its
            # shelf life.
            # ------------------------------------------

            max_age = max(1, min(30, expiry_days // 2))

            manufacture_date = today - timedelta(
                days=random.randint(1, max_age)
            )

            # ------------------------------------------
            # Received date
            #
            # Must be after manufacture date.
            # ------------------------------------------

            days_since_manufacture = (
                today - manufacture_date
            ).days

            received_date = manufacture_date + timedelta(
                days=random.randint(
                    1,
                    max(1, days_since_manufacture)
                )
            )

            # ------------------------------------------
            # Expiry date
            #
            # Must be after received date.
            # ------------------------------------------

            expiry_date = manufacture_date + timedelta(
                days=expiry_days
            )

            # ------------------------------------------
            # Create batch
            # ------------------------------------------

            batch = InventoryBatch(
                inventory_id=inventory.id,
                batch_number=batch_number,
                quantity=quantity,
                manufacture_date=manufacture_date,
                received_date=received_date,
                expiry_date=expiry_date
            )

            db.session.add(batch)

            total_batches += 1

        total_inventory += 1

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    db.session.commit()

    print("\n==============================================")
    print("Inventory setup completed successfully!")
    print("==============================================")

    print("Products processed :", total_inventory)
    print("Inventory records  :", total_inventory)
    print("Batch records      :", total_batches)

    print("\nEach product now has:")
    print("- Current stock")
    print("- 2 inventory batches")
    print("- Manufacture date")
    print("- Received date")
    print("- Expiry date")

    print("\n==============================================")