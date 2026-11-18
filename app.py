
# ================================================================
# SHOPSPHERE 2.1 | MIDNIGHT NEXUS
# AI-POWERED E-COMMERCE INTELLIGENCE
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path

# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="ShopSphere | Midnight Nexus",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CYAN = "#22D3EE"
VIOLET = "#8B5CF6"
BLUE = "#38BDF8"
GREEN = "#34D399"
PINK = "#F472B6"

# ================================================================
# 2. MIDNIGHT NEXUS THEME
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #050914;
    --sidebar: #080F1F;
    --panel: #101D32;
    --cyan: #22D3EE;
    --violet: #8B5CF6;
    --text: #E2E8F0;
    --muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(34,211,238,.09), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(139,92,246,.10), transparent 32%),
        #050914;
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: rgba(5,9,20,.85);
}

.block-container {
    max-width: 100%;
    padding: 2rem 2rem 3rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#070D1B,#0A1426 55%,#080F1D);
    border-right: 1px solid rgba(34,211,238,.22);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem 1rem;
}

[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg,transparent,rgba(34,211,238,.35),transparent);
    margin: 22px 0;
}

[data-testid="stSidebar"] .stRadio > label {
    color: #22D3EE !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 2px;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] .stRadio [role="radio"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 9px 10px;
    transition: all .2s ease;
}

[data-testid="stSidebar"] .stRadio [role="radio"]:hover {
    background: rgba(34,211,238,.08);
    border-color: rgba(34,211,238,.25);
}

[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] {
    background: linear-gradient(100deg,rgba(34,211,238,.16),rgba(139,92,246,.12));
    border: 1px solid rgba(34,211,238,.45);
    box-shadow: 0 0 15px rgba(34,211,238,.08);
}

[data-testid="stSidebar"] .stRadio [role="radio"] p {
    color: #DCEAF5 !important;
    font-size: 13px;
    font-weight: 500;
}

[data-testid="stSidebar"] h3 {
    color: #22D3EE !important;
    font-size: 14px !important;
    letter-spacing: 1.5px;
}

[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #CBD5E1 !important;
    font-size: 12px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #101D32 !important;
    border: 1px solid #263D59 !important;
    border-radius: 9px !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #12364A !important;
    border: 1px solid rgba(34,211,238,.3) !important;
    color: #67E8F9 !important;
    border-radius: 6px !important;
}

/* MAIN HEADINGS */

h1, h2, h3 {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
}

/* HERO */

.hero {
    background: linear-gradient(110deg,#10283D,#171832);
    border: 1px solid rgba(34,211,238,.25);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 0 35px rgba(34,211,238,.06);
}

.hero-title {
    font-size: 29px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #94A3B8;
    margin-top: 8px;
    font-size: 14px;
}

.section-label {
    color: #22D3EE;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin-bottom: 12px;
}

/* KPI CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#101D32,#0A1324);
    border: 1px solid rgba(34,211,238,.22);
    border-radius: 15px;
    padding: 18px;
    min-width: 0;
    overflow: hidden;
    box-shadow: 0 8px 28px rgba(0,0,0,.2);
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {
    color: #22D3EE !important;
    font-size: clamp(17px,1.7vw,27px) !important;
    font-weight: 800 !important;
}

/* CHARTS */

div[data-testid="stPlotlyChart"] {
    background: rgba(11,20,37,.72);
    border: 1px solid rgba(56,189,248,.15);
    border-radius: 16px;
    padding: 10px;
    margin-bottom: 18px;
}

/* INSIGHTS */

.insight-card {
    background: linear-gradient(110deg,#101D32,#0B1425);
    border: 1px solid #263852;
    border-left: 3px solid #22D3EE;
    border-radius: 12px;
    padding: 19px;
    margin: 13px 0;
    color: #DCE6F2;
    line-height: 1.8;
    overflow-wrap: anywhere;
}

.recommendation-card {
    background: linear-gradient(110deg,#17152E,#0B1425);
    border: 1px solid #39305E;
    border-left: 3px solid #8B5CF6;
    border-radius: 12px;
    padding: 19px;
    margin: 13px 0;
    line-height: 1.8;
    overflow-wrap: anywhere;
}

/* BUTTONS */

.stButton button,
.stDownloadButton button {
    background: linear-gradient(100deg,#0891B2,#6D28D9);
    color: white;
    border: 1px solid #28506A;
    border-radius: 10px;
    font-weight: 600;
    transition: .2s ease;
}

.stButton button:hover,
.stDownloadButton button:hover {
    border: 1px solid #22D3EE;
    box-shadow: 0 0 18px rgba(34,211,238,.2);
    color: white;
}

hr {
    border-color: rgba(56,189,248,.18);
}

.sidebar-status {
    background: linear-gradient(120deg,#10263B,#11162D);
    border: 1px solid rgba(34,211,238,.25);
    border-radius: 12px;
    padding: 14px;
    margin-top: 18px;
}

.sidebar-status-title {
    color: #22D3EE;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
}

.sidebar-status-text {
    color: #CBD5E1;
    font-size: 12px;
    margin-top: 8px;
}

@media(max-width:900px) {
    .block-container {
        padding: 1rem;
    }
    .hero-title {
        font-size: 22px;
    }
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# 3. DATA LOADING
# ================================================================

@st.cache_data
def load_main_data():

    path = DATA_DIR / "ecommerce_cleaned.csv"

    if not path.exists():
        return None

    data = pd.read_csv(path)

    data["order_date"] = pd.to_datetime(
        data["order_date"], errors="coerce"
    )

    return data


@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_resource
def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(path)


df = load_main_data()

if df is None:
    st.error(
        "Dataset not found. Place ecommerce_cleaned.csv "
        "inside the data folder."
    )
    st.stop()


# ================================================================
# 4. HELPER FUNCTIONS
# ================================================================

def format_number(value):
    return f"{value:,.2f}"


def show_hero(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_label(title):

    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True
    )


def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1", size=12),
        title_font=dict(color="#E2E8F0", size=17),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", zeroline=False),
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=""
    )

    return fig


def download_csv(data, filename, label):

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


# ================================================================
# 5. FUTURISTIC SIDEBAR
# ================================================================

with st.sidebar:

    # BRANDING

    st.markdown(
        """
        <div style="padding:8px 0 18px 0;">
            <div style="
                font-size:25px;
                font-weight:800;
                color:#22D3EE;
                letter-spacing:-1px;
                white-space:nowrap;
            ">
                🛍️ ShopSphere
            </div>

            <div style="
                font-size:10px;
                color:#8494AC;
                letter-spacing:2px;
                margin-top:8px;
                font-weight:600;
            ">
                MIDNIGHT NEXUS
            </div>

            <div style="
                font-size:11px;
                color:#64748B;
                margin-top:4px;
            ">
                E-COMMERCE INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # NAVIGATION

    pages = {
        "Overview": "◈",
        "Sales Analytics": "↗",
        "Customer Segmentation": "◎",
        "Product Recommendations": "✦",
        "Return Prediction": "⌁",
        "AI Business Insights": "✧"
    }

    section_label("01 / NAVIGATION")

    page = st.radio(
        "EXPLORE MODULES",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]}   {x}",
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.divider()

    # GLOBAL FILTERS

    section_label("02 / GLOBAL FILTERS")

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    date_range = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="global_date_filter"
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    categories = sorted(
        df["category"].dropna().unique()
    )

    selected_categories = st.multiselect(
        "Product categories",
        categories,
        default=categories,
        key="global_category_filter"
    )

    regions = sorted(
        df["region"].dropna().unique()
    )

    selected_regions = st.multiselect(
        "Regions",
        regions,
        default=regions,
        key="global_region_filter"
    )

    st.divider()

    # STATUS PANEL

    st.markdown(
        """
        <div class="sidebar-status">
            <div class="sidebar-status-title">
                SYSTEM STATUS
            </div>
            <div class="sidebar-status-text">
                <span style="color:#34D399;">●</span>
                Dataset Connected
            </div>
            <div class="sidebar-status-text">
                ⚡ Midnight Nexus Active
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"Total dataset records: {len(df):,}")


# ================================================================
# 6. APPLY GLOBAL FILTERS
# ================================================================

filtered_df = df[
    (df["order_date"].dt.date >= start_date) &
    (df["order_date"].dt.date <= end_date) &
    (df["category"].isin(selected_categories)) &
    (df["region"].isin(selected_regions))
].copy()


# ================================================================
# 7. OVERVIEW
# ================================================================

if page == "Overview":

    show_hero(
        "⚡ ShopSphere Intelligence",
        "Your futuristic e-commerce business command center"
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    total_revenue = filtered_df["total_amount"].sum()
    total_orders = filtered_df["order_id"].nunique()
    total_customers = filtered_df["customer_id"].nunique()

    order_totals = filtered_df.groupby("order_id")["total_amount"].sum()
    average_order_value = order_totals.mean()

    return_rate = (
        filtered_df["returned"].astype(str).str.strip().str.lower()
        .eq("yes").mean() * 100
    )

    section_label("01 / BUSINESS PERFORMANCE")

    c1, c2, c3 = st.columns(3, gap="medium")

    c1.metric("TOTAL REVENUE", format_number(total_revenue))
    c2.metric("TOTAL ORDERS", f"{total_orders:,}")
    c3.metric("TOTAL CUSTOMERS", f"{total_customers:,}")

    c4, c5 = st.columns(2, gap="medium")

    c4.metric("AVERAGE ORDER VALUE", format_number(average_order_value))
    c5.metric("RETURN RATE", f"{return_rate:.2f}%")

    st.divider()

    left, right = st.columns([1.6, 1], gap="large")

    with left:

        st.subheader("Revenue Performance")

        monthly = (
            filtered_df.set_index("order_date")
            .resample("MS")["total_amount"]
            .sum()
            .reset_index()
        )

        fig = px.area(
            monthly,
            x="order_date",
            y="total_amount",
            title="Monthly Revenue",
            markers=True
        )

        fig.update_traces(
            line_color=CYAN,
            fillcolor="rgba(34,211,238,.13)"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Category Contribution")

        category = (
            filtered_df.groupby("category")["total_amount"]
            .sum().reset_index()
        )

        fig = px.pie(
            category,
            names="category",
            values="total_amount",
            hole=.65,
            color_discrete_sequence=[CYAN, VIOLET, BLUE, GREEN, PINK]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    left, right = st.columns(2, gap="large")

    with left:

        st.subheader("Regional Performance")

        region = (
            filtered_df.groupby("region")["total_amount"]
            .sum().reset_index()
            .sort_values("total_amount", ascending=False)
        )

        fig = px.bar(
            region,
            x="region",
            y="total_amount",
            color="total_amount",
            color_continuous_scale=["#101D32", "#0891B2", CYAN]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Payment Method Analysis")

        payment = (
            filtered_df.groupby("payment_method")["order_id"]
            .nunique().reset_index(name="orders")
        )

        fig = px.pie(
            payment,
            names="payment_method",
            values="orders",
            hole=.55,
            color_discrete_sequence=[CYAN, VIOLET, BLUE, GREEN, PINK]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    st.subheader("Recent Orders")

    recent = filtered_df.sort_values(
        "order_date", ascending=False
    ).head(10)

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

    download_csv(
        filtered_df,
        "ShopSphere_Filtered_Data.csv",
        "Download Filtered Dataset"
    )


# ================================================================
# 8. SALES ANALYTICS
# ================================================================

elif page == "Sales Analytics":

    show_hero(
        "📈 Sales Analytics",
        "Explore revenue trends and machine learning predictions"
    )

    daily = load_csv("daily_sales.csv")

    if daily is None:

        st.warning("daily_sales.csv not found.")

    else:

        date_col = next(
            (c for c in ["date", "order_date"] if c in daily.columns),
            None
        )

        sales_col = next(
            (c for c in ["sales", "total_amount", "daily_sales"] if c in daily.columns),
            None
        )

        if date_col and sales_col:

            daily[date_col] = pd.to_datetime(
                daily[date_col], errors="coerce"
            )

            daily = daily.dropna(subset=[date_col])

            daily = daily[
                (daily[date_col].dt.date >= start_date) &
                (daily[date_col].dt.date <= end_date)
            ]

            if not daily.empty:

                c1, c2 = st.columns(2)

                c1.metric(
                    "TOTAL DAILY SALES",
                    format_number(daily[sales_col].sum())
                )

                c2.metric(
                    "AVERAGE DAILY SALES",
                    format_number(daily[sales_col].mean())
                )

                fig = px.area(
                    daily,
                    x=date_col,
                    y=sales_col,
                    title="Daily Revenue Performance"
                )

                fig.update_traces(
                    line_color=CYAN,
                    fillcolor="rgba(34,211,238,.13)"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

                download_csv(
                    daily,
                    "ShopSphere_Daily_Sales.csv",
                    "Download Daily Sales"
                )

            else:
                st.info("No daily sales records in this date range.")

        else:
            st.warning("Could not identify date and sales columns.")

    st.divider()

    predictions = load_csv("sales_predictions.csv")

    if predictions is not None:

        st.subheader("Actual vs Predicted Sales")

        actual_col = next(
            (c for c in ["actual_sales", "actual", "sales"] if c in predictions.columns),
            None
        )

        predicted_col = next(
            (c for c in ["predicted_sales", "predicted", "prediction"] if c in predictions.columns),
            None
        )

        prediction_date_col = next(
            (c for c in ["date", "order_date"] if c in predictions.columns),
            None
        )

        if actual_col and predicted_col and prediction_date_col:

            predictions[prediction_date_col] = pd.to_datetime(
                predictions[prediction_date_col], errors="coerce"
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=predictions[prediction_date_col],
                y=predictions[actual_col],
                name="Actual Sales",
                line=dict(color=CYAN, width=2)
            ))

            fig.add_trace(go.Scatter(
                x=predictions[prediction_date_col],
                y=predictions[predicted_col],
                name="Predicted Sales",
                line=dict(color=VIOLET, width=2)
            ))

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            download_csv(
                predictions,
                "ShopSphere_Sales_Predictions.csv",
                "Download Predictions"
            )

        else:
            st.warning("Check the column names in sales_predictions.csv.")

    metrics = load_csv("sales_model_metrics.csv")

    if metrics is not None:

        st.subheader("Model Evaluation")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.caption(
            "Historical test-set evaluation. Predictions are not guaranteed future sales."
        )


# ================================================================
# 9. CUSTOMER SEGMENTATION
# ================================================================

elif page == "Customer Segmentation":

    show_hero(
        "👥 Customer Intelligence",
        "Discover customer groups and purchasing patterns"
    )

    segments = load_csv("customer_segments.csv")
    profile = load_csv("segment_profile.csv")

    if segments is not None:

        segment_column = next(
            (c for c in ["customer_segment", "cluster"] if c in segments.columns),
            None
        )

        if segment_column:

            st.subheader("Customer Segment Explorer")

            segment_values = sorted(
                segments[segment_column].dropna().unique(),
                key=str
            )

            selected_segment = st.selectbox(
                "Select customer segment",
                segment_values
            )

            result = segments[
                segments[segment_column] == selected_segment
            ]

            st.metric(
                "CUSTOMERS IN SELECTED SEGMENT",
                f"{len(result):,}"
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            download_csv(
                result,
                "ShopSphere_Customer_Segment.csv",
                "Download Selected Segment"
            )

        else:
            st.error("No cluster or customer_segment column found.")

    else:
        st.warning("customer_segments.csv not found.")

    if profile is not None:

        st.divider()
        st.subheader("Segment Performance Profile")

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


# ================================================================
# 10. PRODUCT RECOMMENDATIONS
# ================================================================

elif page == "Product Recommendations":

    show_hero(
        "✦ Smart Product Recommendations",
        "Discover products based on customer purchase history"
    )

    recommendations = load_csv("product_recommendations.csv")

    if recommendations is None:
        recommendations = load_csv("customer_recommendations.csv")

    if recommendations is not None:

        if {"customer_id", "product_id"}.issubset(recommendations.columns):

            customers = sorted(
                recommendations["customer_id"].dropna().unique(),
                key=str
            )

            if customers:

                selected_customer = st.selectbox(
                    "Choose Customer ID",
                    customers
                )

                result = recommendations[
                    recommendations["customer_id"] == selected_customer
                ]

                st.subheader("Recommended Products")

                st.metric("RECOMMENDATIONS FOUND", len(result))

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

                download_csv(
                    result,
                    "ShopSphere_Recommendations.csv",
                    "Download Recommendations"
                )

            else:
                st.info("No customer recommendations available.")

        else:
            st.error(
                "Recommendation file must contain customer_id and product_id."
            )

    else:
        st.warning(
            "Recommendation CSV not found. Check the output filename "
            "in product_recommendation.py."
        )


# ================================================================
# 11. RETURN PREDICTION
# ================================================================

elif page == "Return Prediction":

    show_hero(
        "🔮 Return Risk Intelligence",
        "Estimate return probability for a new order"
    )

    saved_model = load_model("return_prediction_model.pkl")

    if saved_model is None:

        st.warning("Return prediction model not found.")

    else:

        model = (
            saved_model.get("model")
            if isinstance(saved_model, dict)
            else saved_model
        )

        if model is None:

            st.error("Saved model does not contain a model object.")

        else:

            st.subheader("New Order Information")

            with st.form("return_prediction_form"):

                col1, col2 = st.columns(2)

                with col1:

                    price = st.number_input(
                        "Product Price",
                        min_value=0.01,
                        value=100.0
                    )

                    discount = st.slider(
                        "Discount",
                        min_value=0.0,
                        max_value=0.30,
                        value=0.05,
                        step=0.01
                    )

                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=5,
                        value=1
                    )

                    age = st.number_input(
                        "Customer Age",
                        min_value=18,
                        max_value=100,
                        value=30
                    )

                with col2:

                    category = st.selectbox(
                        "Category",
                        sorted(df["category"].dropna().unique())
                    )

                    payment = st.selectbox(
                        "Payment Method",
                        sorted(df["payment_method"].dropna().unique())
                    )

                    region = st.selectbox(
                        "Region",
                        sorted(df["region"].dropna().unique())
                    )

                    gender = st.selectbox(
                        "Customer Gender",
                        sorted(df["customer_gender"].dropna().unique())
                    )

                submitted = st.form_submit_button(
                    "⚡ Predict Return Probability",
                    use_container_width=True
                )

            if submitted:

                input_data = pd.DataFrame([{
                    "price": price,
                    "discount": discount,
                    "quantity": quantity,
                    "customer_age": age,
                    "category": category,
                    "payment_method": payment,
                    "region": region,
                    "customer_gender": gender
                }])

                try:

                    if not hasattr(model, "predict_proba"):

                        st.error(
                            "This model does not support probability predictions."
                        )

                    else:

                        probability_values = model.predict_proba(input_data)[0]
                        classes = list(model.classes_)

                        positive_index = next(
                            (
                                i for i, value in enumerate(classes)
                                if str(value).strip().lower()
                                in ["1", "yes", "true", "returned"]
                            ),
                            None
                        )

                        if positive_index is None:

                            st.error(
                                "Could not identify the returned class "
                                "in the trained model."
                            )

                        else:

                            probability = float(
                                probability_values[positive_index]
                            )

                            prediction = model.predict(input_data)[0]

                            st.divider()
                            st.subheader("Prediction Results")

                            c1, c2 = st.columns(2)

                            c1.metric(
                                "RETURN PROBABILITY",
                                f"{probability * 100:.2f}%"
                            )

                            c2.metric(
                                "PREDICTED CLASS",
                                str(prediction)
                            )

                            st.progress(probability)

                            if probability >= 0.5:
                                st.warning("The model predicts a higher return risk.")
                            else:
                                st.success("The model predicts a lower return risk.")

                            st.caption(
                                "Experimental model prediction, not a guarantee."
                            )

                except Exception as error:

                    st.error(f"Prediction failed: {error}")

                    st.info(
                        "Check that your model uses the same input columns "
                        "and preprocessing as this form."
                    )


# ================================================================
# 12. AI BUSINESS INSIGHTS
# ================================================================

elif page == "AI Business Insights":

    show_hero(
        "🧠 Business Intelligence",
        "Understand your business through data-driven insights"
    )

    insights = load_csv("business_insights.csv")
    recommendations = load_csv("business_recommendations.csv")
    kpis = load_csv("business_kpis.csv")

    if kpis is not None:

        st.subheader("Business Performance")

        st.dataframe(
            kpis,
            use_container_width=True,
            hide_index=True
        )

    if insights is not None:

        st.divider()
        section_label("01 / AUTOMATED ANALYTICS")

        st.subheader("✨ Business Insights")

        if "Insight" in insights.columns:

            for i, row in insights.iterrows():

                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>INSIGHT {i + 1:02d}</b><br>
                        {row["Insight"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.dataframe(
                insights,
                use_container_width=True,
                hide_index=True
            )

        download_csv(
            insights,
            "ShopSphere_Business_Insights.csv",
            "Download Business Insights"
        )

    else:

        st.info("Business insights have not been generated yet.")

    if recommendations is not None:

        st.divider()
        section_label("02 / STRATEGIC ACTIONS")

        st.subheader("🚀 Business Recommendations")

        if "Recommendation" in recommendations.columns:

            for i, row in recommendations.iterrows():

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <b>RECOMMENDATION {i + 1:02d}</b><br>
                        {row["Recommendation"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

        download_csv(
            recommendations,
            "ShopSphere_Business_Recommendations.csv",
            "Download Business Recommendations"
        )

    else:

        st.info("Business recommendations have not been generated yet.")


# ================================================================
# 13. FOOTER
# ================================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#64748B;
        font-size:12px;
        letter-spacing:1.5px;
        line-height:2;
    ">
        <span style="color:#22D3EE;font-weight:800;">
            ⚡ SHOPSPHERE 2.1
        </span>
        <br>
        MIDNIGHT NEXUS | AI-POWERED E-COMMERCE INTELLIGENCE
        <br>
        PYTHON • STREAMLIT • MACHINE LEARNING
    </div>
    """,
    unsafe_allow_html=True
)
