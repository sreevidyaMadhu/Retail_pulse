import os
import pandas as pd

from app import app
from models import db, Product, Sale


# --------------------------------------------------
# Sales history file
# --------------------------------------------------

SALES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ml2",
    "sales_history_2year.csv"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

with app.app_context():

    print("\n==============================================")
    print("RetailPulse - Historical Sales Import")
    print("==============================================\n")


    # --------------------------------------------------
    # 1. Check CSV file
    # --------------------------------------------------

    if not os.path.exists(SALES_FILE):

        print("ERROR: Sales history file not found!")
        print("Expected location:")
        print(SALES_FILE)

        exit()

    print("Loading sales history...")

    df = pd.read_csv(SALES_FILE)

    print("Sales records found:", len(df))


    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    required_columns = [
        "date",
        "product_id",
        "product_name",
        "demand"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing columns:")

        for column in missing_columns:
            print("-", column)

        exit()


    # --------------------------------------------------
    # 3. Load products from database
    # --------------------------------------------------

    products = Product.query.all()

    print("Products in database:", len(products))

    if not products:

        print("ERROR: No products found in database.")
        print("Please run populate_master_data.py first.")

        exit()


    # --------------------------------------------------
    # 4. Create product lookup
    # --------------------------------------------------

    product_lookup = {
        product.name: product
        for product in products
    }


    # --------------------------------------------------
    # 5. Check all CSV products exist
    # --------------------------------------------------

    csv_products = df["product_name"].unique()

    missing_products = [
        name
        for name in csv_products
        if name not in product_lookup
    ]

    if missing_products:

        print("\nERROR: These products are missing from database:")

        for name in missing_products:
            print("-", name)

        exit()

    print("All CSV products matched successfully.")


    # --------------------------------------------------
    # 6. Convert date
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])


    # --------------------------------------------------
    # 7. Validate demand
    # --------------------------------------------------

    if df["demand"].isnull().any():

        print("ERROR: Some demand values are missing.")

        exit()

    if (df["demand"] <= 0).any():

        print("ERROR: Some demand values are zero or negative.")

        exit()


    # --------------------------------------------------
    # 8. Remove ONLY previous historical sales
    # --------------------------------------------------
    #
    # Historical sales have:
    #
    #     batch_id = NULL
    #
    # Current application sales have actual batch IDs.
    #
    # Therefore we remove only batch_id IS NULL.
    # This preserves current FEFO sales.
    # --------------------------------------------------

    print("\nRemoving previous historical sales...")

    deleted_count = db.session.query(Sale).filter(
        Sale.batch_id.is_(None)
    ).delete(
        synchronize_session=False
    )

    db.session.commit()

    print(
        "Historical sales removed:",
        deleted_count
    )


    # --------------------------------------------------
    # 9. Prepare historical sales records
    # --------------------------------------------------

    print("\nPreparing sales records...")

    sales_records = []

    for _, row in df.iterrows():

        product = product_lookup[row["product_name"]]

        sale = {
            "product_id": product.id,
            "sale_date": row["date"].to_pydatetime(),

            # Historical sales are not linked
            # to current physical inventory batches.
            "batch_id": None,

            "quantity": int(row["demand"]),

            "sale_price": float(product.price)
        }

        sales_records.append(sale)


    # --------------------------------------------------
    # 10. Insert historical sales in batches
    # --------------------------------------------------

    print("Importing historical sales...")

    batch_size = 5000

    total_records = len(sales_records)

    for start in range(
        0,
        total_records,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_records
        )

        batch = sales_records[start:end]

        db.session.execute(
            Sale.__table__.insert(),
            batch
        )

        db.session.commit()

        print(
            f"Imported {end:,} / "
            f"{total_records:,} records"
        )


    # --------------------------------------------------
    # 11. Final verification
    # --------------------------------------------------

    sales_count = db.session.query(Sale).count()

    historical_count = db.session.query(Sale).filter(
        Sale.batch_id.is_(None)
    ).count()

    current_sales_count = db.session.query(Sale).filter(
        Sale.batch_id.isnot(None)
    ).count()


    print("\n==============================================")
    print("Historical sales import completed!")
    print("==============================================")

    print(
        "CSV records           :",
        len(df)
    )

    print(
        "Total database sales  :",
        sales_count
    )

    print(
        "Historical sales      :",
        historical_count
    )

    print(
        "Current/FEFO sales    :",
        current_sales_count
    )

    print(
        "Products              :",
        len(csv_products)
    )

    print(
        "Date range            :",
        df["date"].min().date(),
        "to",
        df["date"].max().date()
    )

    print(
        "Historical demand     :",
        int(df["demand"].sum()),
        "units"
    )

    print("\nCurrent inventory was NOT changed.")

    print("==============================================")