# 🛍️ RetailPulse: AI-Powered Demand Forecasting & Inventory Management System

**RetailPulse** is an enterprise-grade AI retail operations platform designed to streamline inventory control, optimize stock replenishment, and prevent waste. By integrating real-time weather data from OpenWeatherMap with machine learning models trained on multi-year sales history, RetailPulse delivers highly accurate product demand forecasts and automates **FEFO (First Expired, First Out)** batch management.

---

## 📋 Table of Contents
1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [Database Schema & ERD](#-database-schema--erd)
4. [Machine Learning Pipeline (`ml` & `ml2`)](#-machine-learning-pipeline-ml--ml2)
5. [Directory Structure](#-directory-structure)
6. [API Routes & Access Control](#-api-routes--access-control)
7. [Getting Started & Installation](#-getting-started--installation)
8. [Database Population & Seeding](#-database-population--seeding)
9. [Running Tests](#-running-tests)
10. [Default Credentials](#-default-credentials)

---

## ✨ Key Features

- **🤖 AI-Driven Demand Prediction**: Uses a 2-year trained **Random Forest Regressor** model to forecast daily product demand based on historical sales trends, day of the week, seasonality, and live weather conditions.
- **☀️ Live Weather Integration**: Connects with OpenWeatherMap API to retrieve temperature, rainfall, humidity, wind speed, and atmospheric pressure, mapping weather sensitivity ("Hot Weather", "Rainy Weather", etc.) to product demand.
- **🏷️ FEFO (First Expired, First Out) Batch Tracking**: Manages inventory down to individual batches (`InventoryBatch`) with manufacture and expiration dates. Sales automatically deduct from the earliest expiring batch first.
- **🚨 Intelligent Inventory Risk Analysis**: Automatically identifies stockout risks, calculates usable vs. expired inventory, estimates shortage quantities, and generates actionable replenishment recommendations.
- **🔐 Role-Based Access Control (RBAC)**:
  - **Manager**: Full administrative control over products, inventory, batch registration, demand forecasts, weather insights, and sales analysis.
  - **Employee**: Operational view for daily stock lookup and point-of-sale recording.
- **📊 Interactive Glassmorphism UI**: Built with Jinja2 templates and custom responsive CSS featuring modern visual aesthetics, live metric cards, risk badges, and clear data tables.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Browser]) -->|HTTP Requests| Flask[Flask Web Application - app.py]
    
    subgraph Auth & Routing
        Flask --> Auth[Flask-Login / RBAC]
        Auth --> ManagerDB[Manager Views]
        Auth --> EmployeeDB[Employee Views]
    end

    subgraph Data & Storage
        Flask -->|SQLAlchemy ORM| Postgres[(PostgreSQL Database)]
    end

    subgraph Intelligence & Services
        Flask -->|City Query| Weather[Weather Service - weather_service.py]
        Weather -->|API Call| OWM[OpenWeatherMap API]
        
        Flask -->|Product & Weather Data| ML[ML Inference - ml2/predict_demand.py]
        ML -->|Loads Model| RFModel[Random Forest Model - random_forest_model_2year.pkl]
        ML -->|Loads Features| Encoder[OneHotEncoder - encoder_2year.pkl]
    end

    ML -->|Predicted Demand| Flask
    Postgres -->|Stock & Batches| RiskEngine[Inventory Risk Engine]
    RiskEngine -->|Usable Stock & Shortages| ManagerDB
```

---

## {DATABASE_SCHEMA} Database Schema & ERD

The database schema is built using SQLAlchemy ORM and supports multi-shop retail operations, itemized inventory batch management, and transaction logging.

```mermaid
erDiagram
    SHOPS ||--o{ USERS : employ
    SHOPS ||--o{ PRODUCTS : owns
    PRODUCTS ||--|| INVENTORY : has
    INVENTORY ||--o{ INVENTORY_BATCHES : contains
    PRODUCTS ||--o{ SALES : generates
    INVENTORY_BATCHES ||--o{ SALES : fulfilled_by

    SHOPS {
        int id PK
        string shop_name
        string city
        float latitude
        float longitude
    }

    USERS {
        int id PK
        int shop_id FK
        string username
        string password_hash
        string role
    }

    PRODUCTS {
        int id PK
        int shop_id FK
        string name
        string category
        float price
        string weather_dependency
    }

    INVENTORY {
        int id PK
        int product_id FK
        int quantity
        datetime last_updated
    }

    INVENTORY_BATCHES {
        int id PK
        int inventory_id FK
        string batch_number
        int quantity
        date manufacture_date
        date expiry_date
        date received_date
    }

    SALES {
        int id PK
        int product_id FK
        int batch_id FK
        int quantity
        float sale_price
        datetime sale_date
    }
```

---

## 🧠 Machine Learning Pipeline (`ml` & `ml2`)

RetailPulse features two ML modules:
- `ml/`: Initial version with 1-year sales dataset and basic feature engineering pipeline.
- `ml2/`: Advanced production module trained on **2 years** of historical data (`ml_training_data_2year.csv`).

### Feature Vector Composition
1. **Temporal Features**: `day_of_week`, `month`, `day`, `is_weekend`
2. **Historical Lag Features**: `previous_day_sales`, `rolling_7day_sales`
3. **Weather Metrics**: `temperature`, `min_temperature`, `max_temperature`, `humidity`, `rainfall`, `wind_speed`, `pressure`
4. **Categorical Features** (One-Hot Encoded): `product_id`, `category`, `weather_dependency`

### Model Inference Flow (`ml2/predict_demand.py`)
```python
# Real-time demand inference execution call
predicted_demand = predict_demand(
    product_id="P001",
    category="Beverage",
    weather_dependency="Hot Weather",
    date=datetime.utcnow().date(),
    previous_day_sales=25.0,
    rolling_7day_sales=22.4,
    temperature=32.5,
    min_temperature=27.0,
    max_temperature=34.0,
    humidity=75,
    rainfall=0.0,
    wind_speed=6.0,
    pressure=1012.0
)
```

---

## 📁 Directory Structure

```
retail_pulse/
│
├── .env                              # Environment variables (Database URL, API keys)
├── app.py                            # Main Flask server application and route handlers
├── models.py                         # SQLAlchemy database models & relationship definitions
├── weather_service.py                # OpenWeatherMap API interface with mock fallback
│
├── populate_master_data.py           # Script: Seeds shops, manager/employee accounts, & products
├── populate_inventory.py            # Script: Populates initial inventory & product batches
├── populate_prices.py               # Script: Updates product prices
├── populate_sales.py                # Script: Generates historical sales transactions
│
├── test_flask_ml.py                  # Integration test: Verifies Flask ML pipeline
├── test_inventory_risk.py            # Test: Validates stock risk calculations & recommendations
├── test_sales_features.py            # Test: Validates rolling window sales features
│
├── ml/                               # Machine Learning V1 (1-Year Training Dataset)
│   ├── create_dataset.py             # Dataset synthesis script
│   ├── ml_dataset.py                 # Dataset generator helpers
│   ├── prepare_xy.py                 # X/y feature matrix prep
│   ├── train_model.py                # Training script for V1 model
│   └── ...
│
├── ml2/                              # Machine Learning V2 (Production 2-Year Engine)
│   ├── predict_demand.py             # Inference pipeline & model wrapper
│   ├── random_forest_model_2year.pkl # Trained Random Forest Regressor model
│   ├── encoder_2year.pkl             # OneHotEncoder for categoricals
│   ├── model_features_2year.json     # Feature list & model metadata
│   ├── train_model2.py               # Training script for 2-year model
│   ├── gradient_boosting2.py         # Gradient boosting evaluation
│   ├── baseline_model2.py            # Baseline benchmark models
│   ├── weather_ablation2.py          # Weather feature ablation experiments
│   └── ...
│
├── templates/                        # Jinja2 HTML View Templates
│   ├── base.html                     # Main layout template
│   ├── login.html                    # User login page
│   ├── manager_dashboard.html        # Manager control panel
│   ├── employee_dashboard.html       # Employee interface
│   ├── demand_prediction.html        # AI demand forecasting view
│   ├── weather_insights.html         # Weather impact analytics
│   ├── inventory.html                # Inventory overview table
│   ├── batches.html                  # Batch tracking view
│   ├── add_batch.html                # Add batch modal/form
│   ├── products.html                 # Product master management
│   ├── add_product.html              # Create product form
│   ├── sales.html                    # Sales log table
│   └── add_sale.html                 # Record sale with FEFO form
│
└── static/                           # Static Web Assets
    └── css/                          # CSS stylesheets & theme design rules
```

---

## 🌐 API Routes & Access Control

| Route Pattern | HTTP Methods | Required Role | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Public | Redirects to login page |
| `/login` | `GET`, `POST` | Public | User authentication endpoint |
| `/logout` | `GET` | Authenticated | Logs out active session |
| `/manager/dashboard` | `GET` | Manager | Main dashboard with weather, risk alerts & sales summary |
| `/employee/dashboard` | `GET` | Employee | Operational dashboard for store employees |
| `/manager/demand` | `GET` | Manager | AI demand forecasting view across store catalog |
| `/manager/weather` | `GET` | Manager | Weather-dependent product demand insights |
| `/manager/inventory` | `GET` | Manager / Employee | View current product stock levels |
| `/manager/inventory/<id>/batches` | `GET` | Manager / Employee | Inspect batches & expiry dates for a product |
| `/manager/inventory/<id>/add-batch` | `GET`, `POST` | Manager / Employee | Add a new inventory batch with expiration date |
| `/manager/products` | `GET` | Manager | View product catalog |
| `/manager/products/add` | `GET`, `POST` | Manager | Register new product along with initial batch |
| `/manager/sales` | `GET` | Manager / Employee | View sales history and total sales value |
| `/manager/sales/add` | `GET`, `POST` | Manager / Employee | Record new sale (applies FEFO batch deduction) |

---

## 🛠️ Getting Started & Installation

### 1. Prerequisites
- **Python 3.9+** installed
- **PostgreSQL** database server installed and running (or a local instance)

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/sreevidyaMadhu/Retail_pulse.git
cd Retail_pulse

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask flask-sqlalchemy flask-login psycopg2-binary requests pandas scikit-learn joblib python-dotenv
```

### 4. Configure Environment Variables
Create or update the `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/retailpulse
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

*(Note: If `OPENWEATHER_API_KEY` is not supplied, `weather_service.py` defaults to mock weather mode).*

---

## 🗄️ Database Population & Seeding

Run the population scripts in order to initialize the database tables, seed products, setup inventory batches, and create sample sales history:

```bash
# 1. Populate master shop, users, and products
python populate_master_data.py

# 2. Populate inventory and batch records
python populate_inventory.py

# 3. Update product prices
python populate_prices.py

# 4. Generate historical sales data for rolling feature analysis
python populate_sales.py
```

### 5. Launch the Server
```bash
python app.py
```
Access the application at `http://127.0.0.1:5000/`.

---

## 🧪 Running Tests

Validate system components using the included test suites:

```bash
# Test Flask & ML Demand Prediction Pipeline
python test_flask_ml.py

# Test Inventory Expiry & Risk Engine
python test_inventory_risk.py

# Test Rolling 7-Day & Previous-Day Sales Feature Engine
python test_sales_features.py
```

---

## 🔑 Default Credentials

After running `populate_master_data.py`, the following demo accounts are created:

| Role | Username | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Manager** | `manager1` | `manager123` | Full Administrative Access |
| **Employee** | `employee1` | `employee123` | Store Operations Access |

---

## 📌 Technical Highlights

- **FEFO Safety Guard**: The sales registration endpoint rolls back transactions (`db.session.rollback()`) if requested quantities exceed usable (non-expired) batch stock.
- **Coalesced Rolling Aggregations**: Uses SQL `coalesce(sum(quantity), 0)` to guarantee safe numerical feature extraction even for newly added products with zero prior sales.
- **Graceful Weather Degradation**: Fallback mock service ensures seamless app operation without crashing if network connectivity to OpenWeatherMap fails.
