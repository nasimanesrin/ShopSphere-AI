
# ============================================================
# SHOPSPHERE - AI BUSINESS INSIGHTS
# Automated E-Commerce Analytics
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)

print("\n========== SHOPSPHERE BUSINESS INSIGHTS ==========")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(df["order_date"])

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 3. BUSINESS KPI ANALYSIS
# ============================================================

total_revenue = df["total_amount"].sum()

total_orders = df["order_id"].nunique()

total_customers = df["customer_id"].nunique()

average_order_value = (
    df.groupby("order_id")["total_amount"].sum().mean()
)

total_products = df["product_id"].nunique()

return_rate = (
    df["returned"].astype(str).str.lower().eq("yes").mean()
    * 100
)

repeat_customers = (
    df.groupby("customer_id")["order_id"].nunique()
    .gt(1).sum()
)

repeat_customer_rate = (
    repeat_customers / total_customers * 100
)

kpis = pd.DataFrame({
    "KPI": [
        "Total Revenue",
        "Total Orders",
        "Total Customers",
        "Average Order Value",
        "Unique Products",
        "Return Rate",
        "Repeat Customer Rate"
    ],
    "Value": [
        total_revenue,
        total_orders,
        total_customers,
        average_order_value,
        total_products,
        return_rate,
        repeat_customer_rate
    ]
})

print("\n========== BUSINESS KPIs ==========")
print(kpis.to_string(index=False))

kpis.to_csv(
    OUTPUT_DIR / "business_kpis.csv",
    index=False
)

# ============================================================
# 4. CATEGORY PERFORMANCE
# ============================================================

category_analysis = (
    df.groupby("category")
    .agg(
        revenue=("total_amount", "sum"),
        orders=("order_id", "nunique"),
        quantity_sold=("quantity", "sum")
    )
    .reset_index()
    .sort_values("revenue", ascending=False)
)

category_analysis.to_csv(
    OUTPUT_DIR / "category_insights.csv",
    index=False
)

best_category = category_analysis.iloc[0]

# ============================================================
# 5. REGION PERFORMANCE
# ============================================================

region_analysis = (
    df.groupby("region")
    .agg(
        revenue=("total_amount", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    )
    .reset_index()
    .sort_values("revenue", ascending=False)
)

region_analysis.to_csv(
    OUTPUT_DIR / "region_insights.csv",
    index=False
)

best_region = region_analysis.iloc[0]

# ============================================================
# 6. PAYMENT METHOD ANALYSIS
# ============================================================

payment_analysis = (
    df.groupby("payment_method")
    .agg(
        orders=("order_id", "nunique"),
        revenue=("total_amount", "sum")
    )
    .reset_index()
    .sort_values("orders", ascending=False)
)

payment_analysis.to_csv(
    OUTPUT_DIR / "payment_insights.csv",
    index=False
)

popular_payment = payment_analysis.iloc[0]

# ============================================================
# 7. CUSTOMER ANALYSIS
# ============================================================

customer_analysis = (
    df.groupby("customer_id")
    .agg(
        total_orders=("order_id", "nunique"),
        total_spending=("total_amount", "sum"),
        average_spending=("total_amount", "mean")
    )
    .reset_index()
)

top_customer = customer_analysis.loc[
    customer_analysis["total_spending"].idxmax()
]

customer_analysis.to_csv(
    OUTPUT_DIR / "customer_insights.csv",
    index=False
)

# ============================================================
# 8. RETURN ANALYSIS
# ============================================================

df["returned_flag"] = (
    df["returned"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("yes")
)

return_analysis = (
    df.groupby("category")
    .agg(
        total_orders=("order_id", "nunique"),
        returned_orders=("returned_flag", "sum")
    )
    .reset_index()
)

return_analysis["return_rate"] = (
    return_analysis["returned_orders"]
    / return_analysis["total_orders"] * 100
)

return_analysis.to_csv(
    OUTPUT_DIR / "return_insights.csv",
    index=False
)

highest_return_category = return_analysis.loc[
    return_analysis["return_rate"].idxmax()
]

# ============================================================
# 9. MONTHLY SALES ANALYSIS
# ============================================================

df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("year_month")["total_amount"]
    .sum()
    .reset_index()
)

monthly_sales.columns = ["month", "revenue"]

monthly_sales.to_csv(
    OUTPUT_DIR / "monthly_business_insights.csv",
    index=False
)

best_month = monthly_sales.loc[
    monthly_sales["revenue"].idxmax()
]

# ============================================================
# 10. AUTOMATIC BUSINESS INSIGHTS
# ============================================================

insights = []

insights.append(
    f"Total revenue generated is {total_revenue:,.2f} "
    f"across {total_orders:,} unique orders."
)

insights.append(
    f"{best_category['category']} is the highest-revenue category, "
    f"generating {best_category['revenue']:,.2f}."
)

insights.append(
    f"{best_region['region']} is the highest-revenue region, "
    f"generating {best_region['revenue']:,.2f}."
)

insights.append(
    f"{popular_payment['payment_method']} is the most frequently "
    f"used payment method by order count."
)

insights.append(
    f"The repeat customer rate is {repeat_customer_rate:.2f}%."
)

insights.append(
    f"The overall return rate is {return_rate:.2f}%."
)

insights.append(
    f"{highest_return_category['category']} has the highest "
    f"observed category return rate at "
    f"{highest_return_category['return_rate']:.2f}%."
)

insights.append(
    f"{best_month['month']} is the highest-revenue month "
    f"in the available dataset."
)

insights.append(
    f"The highest-spending customer is {top_customer['customer_id']} "
    f"with total recorded spending of "
    f"{top_customer['total_spending']:,.2f}."
)

# ============================================================
# 11. BUSINESS RECOMMENDATIONS
# ============================================================

recommendations = []

recommendations.append(
    f"Review inventory and marketing opportunities for "
    f"{best_category['category']}, the leading revenue category."
)

recommendations.append(
    f"Study successful sales patterns in {best_region['region']} "
    f"before planning regional campaigns."
)

recommendations.append(
    f"Investigate the reasons for returns in "
    f"{highest_return_category['category']}."
)

if repeat_customer_rate < 50:
    recommendations.append(
        "Consider customer retention campaigns and loyalty programs."
    )
else:
    recommendations.append(
        "Continue testing retention strategies for repeat customers."
    )

if return_rate > 10:
    recommendations.append(
        "Review product descriptions, quality checks and delivery processes."
    )
else:
    recommendations.append(
        "Continue monitoring return rates by category."
    )

# ============================================================
# 12. SAVE INSIGHTS AND RECOMMENDATIONS
# ============================================================

insights_df = pd.DataFrame({
    "Insight": insights
})

recommendations_df = pd.DataFrame({
    "Recommendation": recommendations
})

insights_df.to_csv(
    OUTPUT_DIR / "business_insights.csv",
    index=False
)

recommendations_df.to_csv(
    OUTPUT_DIR / "business_recommendations.csv",
    index=False
)

# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n========== AUTOMATED BUSINESS INSIGHTS ==========")

for number, insight in enumerate(insights, start=1):
    print(f"{number}. {insight}")

print("\n========== BUSINESS RECOMMENDATIONS ==========")

for number, recommendation in enumerate(
    recommendations, start=1
):
    print(f"{number}. {recommendation}")

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\nGenerated Files:")
print("1. business_kpis.csv")
print("2. category_insights.csv")
print("3. region_insights.csv")
print("4. payment_insights.csv")
print("5. customer_insights.csv")
print("6. return_insights.csv")
print("7. monthly_business_insights.csv")
print("8. business_insights.csv")
print("9. business_recommendations.csv")

print("\nBUSINESS INSIGHTS COMPLETED SUCCESSFULLY!")
