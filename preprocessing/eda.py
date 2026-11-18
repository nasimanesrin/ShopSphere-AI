
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1. PATHS AND CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid")


# =====================================
# 2. LOAD DATA
# =====================================

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(df["order_date"])

print("Dataset Shape:", df.shape)


# =====================================
# 3. BASIC EDA
# =====================================

print("\n========== DATASET OVERVIEW ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== UNIQUE VALUES ==========")
print(df.nunique())


# =====================================
# 4. KPI ANALYSIS
# =====================================

total_revenue = df["total_amount"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["customer_id"].nunique()
average_order_value = total_revenue / total_orders

return_rate = (
    df["returned"].eq("Yes").mean() * 100
)

print("\n========== BUSINESS KPIs ==========")

print("Total Revenue:", round(total_revenue, 2))
print("Total Orders:", total_orders)
print("Total Customers:", total_customers)
print("Average Order Value:", round(average_order_value, 2))
print("Return Rate:", round(return_rate, 2), "%")


# =====================================
# 5. MONTHLY SALES ANALYSIS
# =====================================

monthly_sales = (
    df.groupby(df["order_date"].dt.to_period("M"))["total_amount"]
    .sum()
)

plt.figure(figsize=(12, 5))
monthly_sales.index = monthly_sales.index.astype(str)

monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "monthly_sales.png")
plt.close()


# =====================================
# 6. CATEGORY SALES ANALYSIS
# =====================================

category_sales = (
    df.groupby("category")["total_amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
category_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "category_sales.png")
plt.close()


# =====================================
# 7. PAYMENT METHOD ANALYSIS
# =====================================

payment_counts = df["payment_method"].value_counts()

plt.figure(figsize=(8, 5))
payment_counts.plot(kind="bar")

plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "payment_methods.png")
plt.close()


# =====================================
# 8. CUSTOMER AGE ANALYSIS
# =====================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="customer_age",
    bins=20,
    kde=True
)

plt.title("Customer Age Distribution")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "customer_age.png")
plt.close()


# =====================================
# 9. RETURN ANALYSIS
# =====================================

return_counts = df["returned"].value_counts()

plt.figure(figsize=(7, 5))
return_counts.plot(kind="pie", autopct="%1.1f%%")

plt.title("Order Return Distribution")
plt.ylabel("")

plt.savefig(OUTPUT_DIR / "return_analysis.png")
plt.close()


# =====================================
# 10. DELIVERY ANALYSIS
# =====================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="delivery_time_days",
    bins=20,
    kde=True
)

plt.title("Delivery Time Distribution")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "delivery_analysis.png")
plt.close()


# =====================================
# 11. CUSTOMER PURCHASE ANALYSIS
# =====================================

customer_orders = (
    df.groupby("customer_id")["order_id"]
    .nunique()
)

repeat_customers = (customer_orders > 1).sum()

print("\n========== CUSTOMER PURCHASE ==========")
print("One-time customers:", (customer_orders == 1).sum())
print("Repeat customers:", repeat_customers)

repeat_rate = (
    repeat_customers / len(customer_orders) * 100
)

print("Repeat Customer Rate:", round(repeat_rate, 2), "%")


# =====================================
# 12. CORRELATION ANALYSIS
# =====================================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()


# =====================================
# 13. FINAL REPORT
# =====================================

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", OUTPUT_DIR)
