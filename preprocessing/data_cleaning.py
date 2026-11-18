
from pathlib import Path
import pandas as pd

# =====================================
# 1. FILE PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "ecommerce_sales_34500.csv"
OUTPUT_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"


# =====================================
# 2. LOAD DATA
# =====================================

df = pd.read_csv(INPUT_PATH)

print("Original Dataset Shape:", df.shape)


# =====================================
# 3. STANDARDIZE COLUMN NAMES
# =====================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# =====================================
# 4. CLEAN TEXT COLUMNS
# =====================================

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].str.strip()

# Standardize category text
for column in ["category", "payment_method", "region", "customer_gender"]:
    df[column] = df[column].str.title()

# Standardize return labels
df["returned"] = df["returned"].str.strip().str.title()


# =====================================
# 5. CONVERT DATA TYPES
# =====================================

numeric_columns = [
    "price",
    "discount",
    "quantity",
    "delivery_time_days",
    "total_amount",
    "shipping_cost",
    "profit_margin",
    "customer_age"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)


# =====================================
# 6. CHECK MISSING VALUES
# =====================================

print("\nMissing Values Before Treatment:")
print(df.isnull().sum())

# Do not automatically fill values.
# Review any missing values before deciding how to handle them.


# =====================================
# 7. REMOVE DUPLICATES
# =====================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows Found:", duplicates)

df = df.drop_duplicates()


# =====================================
# 8. VALIDATE DATA
# =====================================

print("\nInvalid Data Checks:")

print("Negative price:", (df["price"] < 0).sum())
print("Invalid quantity:", (df["quantity"] <= 0).sum())
print("Invalid discount:", (~df["discount"].between(0, 1)).sum())
print("Negative delivery days:", (df["delivery_time_days"] < 0).sum())
print("Negative shipping cost:", (df["shipping_cost"] < 0).sum())
print("Invalid customer age:", (~df["customer_age"].between(18, 100)).sum())
print("Invalid dates:", df["order_date"].isna().sum())

print(
    "Unexpected return labels:",
    (~df["returned"].isin(["Yes", "No"])).sum()
)


# =====================================
# 9. FINAL DATA QUALITY REPORT
# =====================================

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

print("\nFinal Data Types:")
print(df.dtypes)


# =====================================
# 10. SAVE CLEANED DATA
# =====================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("Output:", OUTPUT_PATH)
