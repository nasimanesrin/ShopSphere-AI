
# 🛍️ ShopSphere – AI-Powered E-Commerce Intelligence

An advanced E-Commerce Analytics and Machine Learning project built using Python, Pandas, NumPy, Scikit-learn, Plotly, and Streamlit.

ShopSphere transforms raw e-commerce transaction data into meaningful business insights through interactive dashboards, data preprocessing, and predictive analytics.

---

## 🎯 Project Objectives

- Analyse e-commerce sales performance.
- Understand customer purchasing behaviour.
- Identify top-performing products and categories.
- Monitor order returns and delivery performance.
- Predict future sales using Machine Learning.
- Segment customers based on purchasing patterns.
- Recommend relevant products.
- Predict potential product returns.
- Generate automated business insights.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| Streamlit | Interactive dashboard |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |

---

## 📂 Project Structure

```text
ShopSphere/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── ecommerce_sales_34500.csv
│   └── ecommerce_cleaned.csv
│
├── preprocessing/
│   └── data_cleaning.py
│
├── models/
│
└── .streamlit/
    └── config.toml
```

---

## 📊 Dataset Information

Dataset: E-Commerce Sales Transactions

Original dataset contains:

- 34,500 records
- 17 columns
- 7,903 unique customers
- Date range: December 2023 – December 2025

### Dataset Columns

| Column | Description |
|---|---|
| order_id | Unique order identifier |
| customer_id | Customer identifier |
| product_id | Product identifier |
| category | Product category |
| price | Product price |
| discount | Discount value |
| quantity | Purchased quantity |
| payment_method | Payment method |
| order_date | Order date |
| delivery_time_days | Delivery duration |
| region | Customer region |
| returned | Return status |
| total_amount | Order amount |
| shipping_cost | Shipping cost |
| profit_margin | Profit margin field |
| customer_age | Customer age |
| customer_gender | Customer gender |

---

## 🧹 Data Cleaning and Preprocessing

The preprocessing module performs:

1. Dataset loading.
2. Data inspection.
3. Column name standardization.
4. Text cleaning.
5. Missing value handling.
6. Duplicate removal.
7. Data type conversion.
8. Data validation.
9. Feature engineering.
10. Cleaned dataset export.

### Feature Engineering

Additional features include:

- order_year
- order_month
- order_month_name
- order_quarter
- order_day
- order_day_name
- is_weekend
- age_group
- discount_percentage
- gross_item_value
- delivery_category

---

## 🤖 Machine Learning Features

Planned AI and ML modules:

### 1. Sales Prediction
Predict future sales trends using regression algorithms.

### 2. Customer Segmentation
Group customers using K-Means clustering.

### 3. Product Recommendation
Recommend products based on purchasing behaviour.

### 4. Return Prediction
Classify orders according to their likelihood of being returned.

### 5. AI Business Insights
Generate automated explanations of important business trends.

---

## 📈 Dashboard Modules

- Overview Dashboard
- Sales Analytics
- Product Analytics
- Customer Analytics
- Order and Returns Analysis
- Data Explorer
- AI Insights

---

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project

```bash
cd ShopSphere
```

### Step 3: Create a Virtual Environment

```bash
python -m venv env
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run Data Preprocessing

```bash
python preprocessing/data_cleaning.py
```

### Step 7: Launch the Dashboard

```bash
streamlit run app.py
```

---

## 🎨 Dashboard Theme

**Midnight Analytics**

A modern dark-themed dashboard with interactive visualizations, KPI cards, and professional data presentation.

---

## 🚀 Future Enhancements

- Advanced sales forecasting.
- Customer lifetime value prediction.
- Recommendation engine.
- Interactive AI assistant.
- Automated business reports.
- Model performance monitoring.

---

## 👩‍💻 Developed By

**Naseema Nesrin**

B.Tech – Artificial Intelligence and Data Science

Project: ShopSphere – AI-Powered E-Commerce Intelligence

---

## 📌 Project Status

Currently under development.

Built as a portfolio project to demonstrate Data Analytics, Data Preprocessing, Machine Learning, and Dashboard Development skills.
