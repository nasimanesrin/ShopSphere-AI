
# ============================================================
# SHOPSPHERE - RETURN PREDICTION
# Algorithms: Logistic Regression & Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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

print("\n========== SHOPSPHERE RETURN PREDICTION ==========")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

numeric_features = [
    "price",
    "discount",
    "quantity",
    "customer_age"
]

categorical_features = [
    "category",
    "payment_method",
    "region",
    "customer_gender"
]

target = "returned"

required_columns = (
    numeric_features + categorical_features + [target]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df[required_columns].copy()

df = df.dropna(subset=[target])

# Normalize target values
df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
}

df[target] = df[target].map(target_mapping)

df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

X = df[numeric_features + categorical_features]
y = df[target]

print("\nTarget distribution:")
print(y.value_counts())

print("\nReturn percentage:")
print((y.mean() * 100).round(2), "%")

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[model_name] = pipeline

    print(f"\n{model_name} Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

# ============================================================
# 8. SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "return_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Select model using F1 score
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# ============================================================
# 9. FINAL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

prediction_results = X_test.copy()

prediction_results["actual_returned"] = y_test.values
prediction_results["predicted_returned"] = best_predictions
prediction_results["return_probability"] = best_probabilities

prediction_results.to_csv(
    OUTPUT_DIR / "return_predictions.csv",
    index=False
)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Not Returned", "Returned"],
    yticklabels=["Not Returned", "Returned"]
)

plt.title(f"Return Prediction - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "return_confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# 11. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

rf_pipeline = trained_models["Random Forest"]

rf_classifier = rf_pipeline.named_steps["classifier"]

feature_names = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_classifier.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "return_feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 7))

top_features = importance.head(15).sort_values(
    "Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Features - Return Prediction")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "return_feature_importance.png",
    dpi=300
)

plt.close()

# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target
    },
    MODEL_DIR / "return_prediction_model.pkl"
)

print("\nBest model saved successfully!")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========== RETURN PREDICTION SUMMARY ==========")

print("Total records:", len(df))
print("Returned orders:", int(y.sum()))
print("Not returned:", int((y == 0).sum()))
print("Best model:", best_model_name)

print("\nGenerated Files:")
print("1. return_model_comparison.csv")
print("2. return_predictions.csv")
print("3. return_feature_importance.csv")
print("4. return_confusion_matrix.png")
print("5. return_feature_importance.png")
print("6. models/return_prediction_model.pkl")

print("\nRETURN PREDICTION COMPLETED!")
