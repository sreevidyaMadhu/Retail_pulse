
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class Shop(db.Model):
    __tablename__ = 'shops'

    id = db.Column(db.Integer, primary_key=True)
    shop_name = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    # Relationships
    users = db.relationship(
        'User',
        backref='shop',
        lazy=True,
        cascade='all, delete-orphan'
    )

    products = db.relationship(
        'Product',
        backref='shop',
        lazy=True,
        cascade='all, delete-orphan'
    )


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)

    shop_id = db.Column(
        db.Integer,
        db.ForeignKey('shops.id'),
        nullable=False
    )

    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    # manager or employee
    role = db.Column(db.String(20), nullable=False)


class Product(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True)

    shop_id = db.Column(
        db.Integer,
        db.ForeignKey('shops.id'),
        nullable=False
    )

    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0)

    weather_dependency = db.Column(db.String(50))

    # One product → one inventory record
    inventory = db.relationship(
        'Inventory',
        backref='product',
        uselist=False,
        cascade='all, delete-orphan'
    )

    # One product → many sales
    sales = db.relationship(
        'Sale',
        back_populates='product',
        lazy=True
    )

class Inventory(db.Model):
    __tablename__ = 'inventory'

    id = db.Column(db.Integer, primary_key=True)

    product_id = db.Column(
        db.Integer,
        db.ForeignKey('products.id'),
        nullable=False,
        unique=True
    )

    quantity = db.Column(db.Integer, nullable=False, default=0)

    reorder_level = db.Column(
        db.Integer,
        nullable=False,
        default=10
    )

    last_updated = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
class InventoryBatch(db.Model):
    __tablename__ = "inventory_batches"

    id = db.Column(db.Integer, primary_key=True)

    inventory_id = db.Column(
        db.Integer,
        db.ForeignKey("inventory.id"),
        nullable=False
    )

    batch_number = db.Column(
        db.String(50),
        nullable=False
    )

    quantity = db.Column(
        db.Integer,
        nullable=False,
        default=0
    )

    manufacture_date = db.Column(
        db.Date,
        nullable=True
    )

    expiry_date = db.Column(
        db.Date,
        nullable=False
    )

    received_date = db.Column(
        db.Date,
        nullable=False
    )

    # One batch → many sales
    sales = db.relationship(
        "Sale",
        back_populates="batch",
        lazy=True
    )
class Sale(db.Model):
    __tablename__ = "sales"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    product_id = db.Column(
        db.Integer,
        db.ForeignKey("products.id"),
        nullable=False
    )

    batch_id = db.Column(
        db.Integer,
        db.ForeignKey("inventory_batches.id"),
        nullable=True
    )

    quantity = db.Column(
        db.Integer,
        nullable=False
    )

    sale_price = db.Column(
        db.Float,
        nullable=True
    )

    sale_date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    product = db.relationship(
        "Product",
        back_populates="sales"
    )

    batch = db.relationship(
        "InventoryBatch",
        back_populates="sales"
    )