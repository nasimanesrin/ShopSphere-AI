
# ============================================================
# SHOPSPHERE - SALES PREDICTION USING MACHINE LEARNING
# Algorithm: Random Forest Regressor
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ecommerce_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n========== SHOPSPHERE SALES PREDICTION ==========")

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(df["order_date"])

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 3. DAILY SALES AGGREGATION
# ============================================================

daily_sales = (
    df.groupby("order_date")["total_amount"]
    .sum()
    .reset_index()
)

daily_sales.columns = ["date", "sales"]

# Fill missing calendar dates with zero sales
daily_sales = (
    daily_sales.set_index("date")
    .asfreq("D", fill_value=0)
    .rename_axis("date")
    .reset_index()
)

daily_sales.to_csv(
    OUTPUT_DIR / "daily_sales.csv",
    index=False
)

print("\nDaily sales generated!")
print(daily_sales.head())

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data["day_of_week"] = data["date"].dt.dayofweek
    data["quarter"] = data["date"].dt.quarter
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Historical sales features
    data["lag_1"] = data["sales"].shift(1)
    data["lag_7"] = data["sales"].shift(7)
    data["lag_14"] = data["sales"].shift(14)
    data["lag_30"] = data["sales"].shift(30)

    # Rolling averages using past days only
    data["rolling_mean_7"] = (
        data["sales"].shift(1).rolling(7).mean()
    )

    data["rolling_mean_30"] = (
        data["sales"].shift(1).rolling(30).mean()
    )

    return data


featured_data = create_features(daily_sales)

featured_data = featured_data.dropna().reset_index(drop=True)

FEATURE_COLUMNS = [
    "day",
    "month",
    "year",
    "day_of_week",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

X = featured_data[FEATURE_COLUMNS]
y = featured_data["sales"]

print("\nFeature engineering completed!")
print("Feature dataset shape:", featured_data.shape)

# ============================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split_index = int(len(featured_data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 6. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ============================================================
# 7. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# 8. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(
    OUTPUT_DIR / "sales_model_metrics.csv",
    index=False
)

# ============================================================
# 9. ACTUAL VS PREDICTED DATA
# ============================================================

results = pd.DataFrame({
    "date": featured_data["date"].iloc[split_index:].values,
    "actual_sales": y_test.values,
    "predicted_sales": y_pred
})

results.to_csv(
    OUTPUT_DIR / "sales_predictions.csv",
    index=False
)

# ============================================================
# 10. ACTUAL VS PREDICTED CHART
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    results["date"],
    results["actual_sales"],
    label="Actual Sales"
)

plt.plot(
    results["date"],
    results["predicted_sales"],
    label="Predicted Sales"
)

plt.title("ShopSphere - Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_sales.png",
    dpi=300
)

plt.close()

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "sales_feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Sales Prediction - Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sales_feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# 12. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": FEATURE_COLUMNS
    },
    MODEL_DIR / "sales_prediction_model.pkl"
)

print("\nModel saved successfully!")

# ============================================================
# 13. SUMMARY
# ============================================================

print("\n========== SALES PREDICTION SUMMARY ==========")

print("Total daily records:", len(daily_sales))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nGenerated Files:")
print("1. daily_sales.csv")
print("2. sales_model_metrics.csv")
print("3. sales_predictions.csv")
print("4. sales_feature_importance.csv")
print("5. actual_vs_predicted_sales.png")
print("6. sales_feature_importance.png")
print("7. models/sales_prediction_model.pkl")

print("\nSALES PREDICTION COMPLETED SUCCESSFULLY!")
