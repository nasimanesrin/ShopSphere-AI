
# ============================================================
# SHOPSPHERE - PRODUCT RECOMMENDATION SYSTEM
# Algorithm: Item-Based Collaborative Filtering
# Similarity: Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("\n========== SHOPSPHERE PRODUCT RECOMMENDATION ==========")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

required_columns = [
    "customer_id",
    "product_id",
    "quantity"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df = df.dropna(subset=required_columns)

print("Dataset loaded:", df.shape)

# ============================================================
# 3. CREATE CUSTOMER-PRODUCT INTERACTIONS
# ============================================================

interactions = (
    df.groupby(["customer_id", "product_id"])["quantity"]
    .sum()
    .reset_index()
)

print("Customer-product interactions:", len(interactions))

# Convert customer and product IDs into numeric indices
customer_codes, customer_ids = pd.factorize(
    interactions["customer_id"]
)

product_codes, product_ids = pd.factorize(
    interactions["product_id"]
)

# ============================================================
# 4. CREATE SPARSE CUSTOMER-PRODUCT MATRIX
# ============================================================

interaction_matrix = csr_matrix(
    (
        interactions["quantity"].astype(float),
        (customer_codes, product_codes)
    ),
    shape=(len(customer_ids), len(product_ids))
)

print("Customer matrix shape:", interaction_matrix.shape)

# ============================================================
# 5. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================

# Each row now represents a product and its customer interactions
item_customer_matrix = interaction_matrix.T.tocsr()

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=min(11, len(product_ids))
)

model.fit(item_customer_matrix)

print("Recommendation model trained!")

# ============================================================
# 6. GENERATE RECOMMENDATIONS
# ============================================================

def recommend_products(customer_id, top_n=5):

    customer_positions = np.where(
        customer_ids == customer_id
    )[0]

    if len(customer_positions) == 0:
        return pd.DataFrame(
            columns=["customer_id", "recommended_product_id", "similarity_score"]
        )

    customer_index = customer_positions[0]

    # Products already purchased by this customer
    purchased_products = interaction_matrix[
        customer_index
    ].indices

    if len(purchased_products) == 0:
        return pd.DataFrame(
            columns=["customer_id", "recommended_product_id", "similarity_score"]
        )

    purchased_quantities = interaction_matrix[
        customer_index, purchased_products
    ].toarray().ravel()

    # Find similar products for all purchased products
    distances, neighbors = model.kneighbors(
        item_customer_matrix[purchased_products]
    )

    scores = np.zeros(len(product_ids))

    # Weight similarity by customer's purchase quantity
    for i, source_product in enumerate(purchased_products):

        similarities = 1 - distances[i]

        for neighbor, similarity in zip(
            neighbors[i], similarities
        ):

            scores[neighbor] += (
                similarity * purchased_quantities[i]
            )

    # Do not recommend products already purchased
    scores[purchased_products] = 0

    # Get top recommendations
    recommended_indices = np.argsort(scores)[::-1]

    recommended_indices = [
        index for index in recommended_indices
        if scores[index] > 0
    ][:top_n]

    recommendations = pd.DataFrame({
        "customer_id": customer_id,
        "recommended_product_id": [
            product_ids[index]
            for index in recommended_indices
        ],
        "similarity_score": [
            round(scores[index], 4)
            for index in recommended_indices
        ]
    })

    return recommendations


# ============================================================
# 7. GENERATE RECOMMENDATIONS FOR ALL CUSTOMERS
# ============================================================

all_recommendations = []

print("\nGenerating customer recommendations...")

for customer_id in customer_ids:

    customer_recommendations = recommend_products(
        customer_id,
        top_n=5
    )

    if not customer_recommendations.empty:
        all_recommendations.append(customer_recommendations)

if all_recommendations:
    recommendation_df = pd.concat(
        all_recommendations,
        ignore_index=True
    )
else:
    recommendation_df = pd.DataFrame(
        columns=[
            "customer_id",
            "recommended_product_id",
            "similarity_score"
        ]
    )

# ============================================================
# 8. SAVE RECOMMENDATIONS
# ============================================================

recommendation_path = (
    OUTPUT_DIR / "product_recommendations.csv"
)

recommendation_df.to_csv(
    recommendation_path,
    index=False
)

print("\nRecommendations saved!")
print(recommendation_df.head(10))

# ============================================================
# 9. SAVE PRODUCT POPULARITY
# ============================================================

product_popularity = (
    interactions.groupby("product_id")
    .agg(
        total_quantity=("quantity", "sum"),
        unique_customers=("customer_id", "nunique")
    )
    .reset_index()
    .sort_values(
        "total_quantity",
        ascending=False
    )
)

product_popularity.to_csv(
    OUTPUT_DIR / "product_popularity.csv",
    index=False
)

# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "customer_ids": customer_ids,
        "product_ids": product_ids,
        "interaction_matrix": interaction_matrix
    },
    MODEL_DIR / "product_recommendation_model.pkl"
)

print("\nModel saved successfully!")

# ============================================================
# 11. TEST A CUSTOMER
# ============================================================

sample_customer = customer_ids[0]

print("\nSample Customer:", sample_customer)

sample_recommendations = recommend_products(
    sample_customer,
    top_n=5
)

print("\nRecommended Products:")
print(sample_recommendations)

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n========== RECOMMENDATION SUMMARY ==========")

print("Total customers:", len(customer_ids))
print("Total products:", len(product_ids))
print("Total interactions:", len(interactions))
print("Generated recommendations:", len(recommendation_df))

print("\nGenerated Files:")
print("1. product_recommendations.csv")
print("2. product_popularity.csv")
print("3. models/product_recommendation_model.pkl")

print("\nPRODUCT RECOMMENDATION COMPLETED!")