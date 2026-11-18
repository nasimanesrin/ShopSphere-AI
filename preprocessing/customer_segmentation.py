
# ============================================================
# SHOPSPHERE - AI POWERED E-COMMERCE INTELLIGENCE
# CUSTOMER SEGMENTATION USING K-MEANS
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = OUTPUT_DIR / "customer_features.csv"
SCALED_PATH = OUTPUT_DIR / "customer_features_scaled.csv"

SEGMENT_PATH = OUTPUT_DIR / "customer_segments.csv"
PROFILE_PATH = OUTPUT_DIR / "segment_profile.csv"

SCALER_PATH = OUTPUT_DIR / "customer_scaler.pkl"
MODEL_PATH = OUTPUT_DIR / "customer_kmeans_model.pkl"

ELBOW_PATH = OUTPUT_DIR / "elbow_method.png"
SILHOUETTE_PATH = OUTPUT_DIR / "silhouette_scores.png"

DISTRIBUTION_PATH = OUTPUT_DIR / "customer_segment_distribution.png"
SPENDING_PATH = OUTPUT_DIR / "segment_spending_comparison.png"
ORDERS_PATH = OUTPUT_DIR / "segment_order_comparison.png"
AOV_PATH = OUTPUT_DIR / "segment_aov_comparison.png"
SCATTER_PATH = OUTPUT_DIR / "customer_spending_vs_orders.png"


# ============================================================
# 3. LOAD CLEANED DATASET
# ============================================================

print("\n" + "=" * 65)
print("SHOPSPHERE - CUSTOMER SEGMENTATION")
print("=" * 65)

print("\n[STEP 1] LOADING DATASET")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[STEP 2] CHECKING REQUIRED COLUMNS")

required_columns = [
    "customer_id",
    "order_id",
    "total_amount",
    "quantity",
    "discount",
    "delivery_time_days"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("All required columns are available!")


# ============================================================
# 5. CUSTOMER-LEVEL AGGREGATION
# ============================================================

print("\n[STEP 3] CUSTOMER-LEVEL AGGREGATION")

customer_df = df.groupby("customer_id").agg(

    total_orders=("order_id", "nunique"),

    total_spending=("total_amount", "sum"),

    average_order_value=("total_amount", "mean"),

    total_quantity=("quantity", "sum"),

    average_discount=("discount", "mean"),

    average_delivery_days=("delivery_time_days", "mean")

).reset_index()

print("Customer Features Created Successfully!")
print("Customer Dataset Shape:", customer_df.shape)

print("\nCustomer Data Preview:")
print(customer_df.head())


# ============================================================
# 6. DATA VALIDATION
# ============================================================

print("\n[STEP 4] DATA VALIDATION")

print("\nMissing Values:")
print(customer_df.isnull().sum())

if customer_df.isnull().any().any():
    raise ValueError("Missing values found in customer features.")

numeric_values = customer_df.select_dtypes(
    include=np.number
).to_numpy()

if not np.isfinite(numeric_values).all():
    raise ValueError("Infinite values found in customer features.")

print("\nData Validation Completed!")


# ============================================================
# 7. SAVE CUSTOMER FEATURES
# ============================================================

customer_df.to_csv(FEATURE_PATH, index=False)

print("\nCustomer Features Saved:", FEATURE_PATH)


# ============================================================
# 8. FEATURE SELECTION
# ============================================================

print("\n[STEP 5] FEATURE SELECTION")

features = [
    "total_orders",
    "total_spending",
    "average_order_value",
    "total_quantity",
    "average_discount",
    "average_delivery_days"
]

X = customer_df[features].copy()

print("Selected Features:", features)
print("Feature Dataset Shape:", X.shape)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

print("\n[STEP 6] FEATURE SCALING")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\nScaled Dataset Preview:")
print(scaled_df.head())

print("Scaled Dataset Shape:", scaled_df.shape)


# ============================================================
# 10. SAVE SCALER AND SCALED DATA
# ============================================================

scaled_df.to_csv(SCALED_PATH, index=False)

joblib.dump(scaler, SCALER_PATH)

print("\nScaler Saved Successfully!")


# ============================================================
# 11. ELBOW METHOD AND SILHOUETTE SCORE
# ============================================================

print("\n[STEP 7] FINDING OPTIMAL NUMBER OF CLUSTERS")

inertia_values = []
silhouette_values = []

K_RANGE = range(2, 11)

for k in K_RANGE:

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans_test.fit_predict(X_scaled)

    inertia = kmeans_test.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f} | "
        f"Silhouette Score = {silhouette:.4f}"
    )


# ============================================================
# 12. ELBOW METHOD VISUALIZATION
# ============================================================

print("\n[STEP 8] GENERATING ELBOW GRAPH")

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_RANGE),
    inertia_values,
    marker="o",
    linewidth=2
)

plt.title("Elbow Method - Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.xticks(list(K_RANGE))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ELBOW_PATH, dpi=300)
plt.close()

print("Elbow Graph Saved:", ELBOW_PATH)


# ============================================================
# 13. SILHOUETTE SCORE VISUALIZATION
# ============================================================

print("\n[STEP 9] GENERATING SILHOUETTE GRAPH")

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_RANGE),
    silhouette_values,
    marker="o",
    linewidth=2
)

plt.title("Silhouette Score - Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(list(K_RANGE))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(SILHOUETTE_PATH, dpi=300)
plt.close()

print("Silhouette Graph Saved:", SILHOUETTE_PATH)


# ============================================================
# 14. SELECT NUMBER OF CLUSTERS
# ============================================================

print("\n[STEP 10] SELECTING NUMBER OF CLUSTERS")

# Temporary choice.
# Review the Elbow and Silhouette graphs before finalizing.

OPTIMAL_K = 3

print("Selected Number of Clusters:", OPTIMAL_K)


# ============================================================
# 15. TRAIN FINAL K-MEANS MODEL
# ============================================================

print("\n[STEP 11] TRAINING FINAL K-MEANS MODEL")

kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

customer_df["cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means Model Trained Successfully!")


# ============================================================
# 16. CUSTOMER SEGMENT DISTRIBUTION
# ============================================================

print("\n[STEP 12] CUSTOMER SEGMENT DISTRIBUTION")

segment_counts = (
    customer_df["cluster"]
    .value_counts()
    .sort_index()
)

segment_percentage = (
    customer_df["cluster"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\nCustomers in Each Cluster:")
print(segment_counts)

print("\nCustomer Percentage:")
print(segment_percentage.round(2))


# ============================================================
# 17. CUSTOMER SEGMENT PROFILE
# ============================================================

print("\n[STEP 13] CUSTOMER SEGMENT PROFILE")

segment_profile = customer_df.groupby("cluster").agg(

    customer_count=("customer_id", "count"),

    average_orders=("total_orders", "mean"),

    average_spending=("total_spending", "mean"),

    average_order_value=("average_order_value", "mean"),

    average_quantity=("total_quantity", "mean"),

    average_discount=("average_discount", "mean"),

    average_delivery_days=("average_delivery_days", "mean")

).round(2)

print("\nCustomer Segment Profile:")
print(segment_profile)


# ============================================================
# 18. ASSIGN BUSINESS-FRIENDLY SEGMENT LABELS
# ============================================================

print("\n[STEP 14] ASSIGNING CUSTOMER SEGMENT LABELS")

sorted_clusters = (
    segment_profile["average_spending"]
    .sort_values()
    .index.tolist()
)

if OPTIMAL_K == 3:

    segment_names = {
        sorted_clusters[0]: "Low-Spending Customers",
        sorted_clusters[1]: "Mid-Spending Customers",
        sorted_clusters[2]: "High-Spending Customers"
    }

else:

    segment_names = {
        cluster_id: f"Spending Segment {rank}"
        for rank, cluster_id in enumerate(
            sorted_clusters, start=1
        )
    }

customer_df["customer_segment"] = (
    customer_df["cluster"].map(segment_names)
)

segment_profile["customer_segment"] = (
    segment_profile.index.map(segment_names)
)

print("\nAssigned Segment Labels:")
print(segment_names)

print("\nUpdated Segment Profile:")
print(segment_profile)


# ============================================================
# 19. BUSINESS INTERPRETATION
# ============================================================

print("\n[STEP 15] BUSINESS INTERPRETATION")

for cluster_id, row in segment_profile.iterrows():

    print("\n" + "-" * 50)

    print("Segment:", row["customer_segment"])
    print("Customer Count:", int(row["customer_count"]))
    print("Average Spending:", row["average_spending"])
    print("Average Orders:", row["average_orders"])
    print("Average Order Value:", row["average_order_value"])
    print("Average Quantity:", row["average_quantity"])
    print("Average Discount:", row["average_discount"])
    print("Average Delivery Days:", row["average_delivery_days"])


# ============================================================
# 20. SAVE FINAL CUSTOMER SEGMENTS
# ============================================================

print("\n[STEP 16] SAVING CUSTOMER SEGMENTS")

customer_df.to_csv(SEGMENT_PATH, index=False)

segment_profile.to_csv(PROFILE_PATH)

print("Customer Segments Saved:", SEGMENT_PATH)
print("Segment Profile Saved:", PROFILE_PATH)


# ============================================================
# 21. SAVE TRAINED MODELS
# ============================================================

print("\n[STEP 17] SAVING TRAINED MODELS")

joblib.dump(kmeans, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("K-Means Model Saved:", MODEL_PATH)
print("Scaler Saved:", SCALER_PATH)


# ============================================================
# 22. CUSTOMER SEGMENT DISTRIBUTION CHART
# ============================================================

print("\n[STEP 18] CUSTOMER SEGMENT DISTRIBUTION")

chart_labels = [
    segment_names[cluster_id]
    for cluster_id in segment_counts.index
]

plt.figure(figsize=(10, 6))

plt.bar(
    chart_labels,
    segment_counts.values
)

plt.title("Customer Distribution by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(DISTRIBUTION_PATH, dpi=300)
plt.close()


# ============================================================
# 23. AVERAGE CUSTOMER SPENDING COMPARISON
# ============================================================

print("\n[STEP 19] CUSTOMER SPENDING COMPARISON")

plt.figure(figsize=(10, 6))

plt.bar(
    segment_profile["customer_segment"],
    segment_profile["average_spending"]
)

plt.title("Average Customer Spending by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Spending")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(SPENDING_PATH, dpi=300)
plt.close()


# ============================================================
# 24. AVERAGE ORDERS COMPARISON
# ============================================================

print("\n[STEP 20] CUSTOMER ORDER COMPARISON")

plt.figure(figsize=(10, 6))

plt.bar(
    segment_profile["customer_segment"],
    segment_profile["average_orders"]
)

plt.title("Average Orders by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Number of Orders")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(ORDERS_PATH, dpi=300)
plt.close()


# ============================================================
# 25. AVERAGE ORDER VALUE COMPARISON
# ============================================================

print("\n[STEP 21] AVERAGE ORDER VALUE COMPARISON")

plt.figure(figsize=(10, 6))

plt.bar(
    segment_profile["customer_segment"],
    segment_profile["average_order_value"]
)

plt.title("Average Order Value by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Order Value")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(AOV_PATH, dpi=300)
plt.close()


# ============================================================
# 26. CUSTOMER SPENDING VS ORDER FREQUENCY
# ============================================================

print("\n[STEP 22] SPENDING VS ORDER FREQUENCY")

plt.figure(figsize=(10, 6))

for cluster_id, group in customer_df.groupby("cluster"):

    plt.scatter(
        group["total_orders"],
        group["total_spending"],
        label=segment_names[cluster_id],
        alpha=0.6
    )

plt.title("Customer Spending vs Order Frequency")
plt.xlabel("Total Orders")
plt.ylabel("Total Spending")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(SCATTER_PATH, dpi=300)
plt.close()


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("Total Orders:", df["order_id"].nunique())
print("Total Customers:", len(customer_df))
print("Total Features:", len(features))
print("Number of Clusters:", OPTIMAL_K)

print("\nGENERATED FILES:")

print("1. customer_features.csv")
print("2. customer_features_scaled.csv")
print("3. customer_segments.csv")
print("4. segment_profile.csv")
print("5. customer_scaler.pkl")
print("6. customer_kmeans_model.pkl")
print("7. elbow_method.png")
print("8. silhouette_scores.png")
print("9. customer_segment_distribution.png")
print("10. segment_spending_comparison.png")
print("11. segment_order_comparison.png")
print("12. segment_aov_comparison.png")
print("13. customer_spending_vs_orders.png")

print("\nALL CUSTOMER SEGMENTATION TASKS COMPLETED!")
