
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "ecommerce_transformed.csv"


# ==========================================
# 2. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv(INPUT_PATH)

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

print("Cleaned Dataset Shape:", df.shape)


# ==========================================
# 3. DATE TRANSFORMATION
# ==========================================

df["order_year"] = df["order_date"].dt.year

df["order_month"] = df["order_date"].dt.month

df["order_month_name"] = df["order_date"].dt.month_name()

df["order_quarter"] = df["order_date"].dt.quarter

df["order_day"] = df["order_date"].dt.day

df["order_day_name"] = df["order_date"].dt.day_name()

df["is_weekend"] = (
    df["order_date"].dt.dayofweek >= 5
).astype(int)


# ==========================================
# 4. NUMERICAL FEATURE ENGINEERING
# ==========================================

# Gross item value before discount
df["gross_item_value"] = (
    df["price"] * df["quantity"]
)

# Discount percentage
df["discount_percentage"] = (
    df["discount"] * 100
)

# Total discount amount
df["discount_amount"] = (
    df["gross_item_value"] * df["discount"]
)


# ==========================================
# 5. CUSTOMER AGE GROUPS
# ==========================================

df["age_group"] = pd.cut(
    df["customer_age"],
    bins=[17, 25, 35, 45, 55, 70, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-70",
        "71-100"
    ]
)


# ==========================================
# 6. DELIVERY CATEGORIES
# ==========================================

df["delivery_category"] = pd.cut(
    df["delivery_time_days"],
    bins=[-1, 3, 5, 7, np.inf],
    labels=[
        "Fast",
        "Standard",
        "Slow",
        "Very Slow"
    ]
)


# ==========================================
# 7. RETURN STATUS ENCODING
# ==========================================

df["returned_binary"] = (
    df["returned"]
    .map({"No": 0, "Yes": 1})
)


# ==========================================
# 8. CATEGORICAL DATA ENCODING
# ==========================================

categorical_columns = [
    "category",
    "payment_method",
    "region",
    "customer_gender"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)


# ==========================================
# 9. FINAL DATA INSPECTION
# ==========================================

print("\n========== TRANSFORMED DATA ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 10. SAVE TRANSFORMED DATA
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nTransformation Completed Successfully!")
print("Saved File:", OUTPUT_PATH)
