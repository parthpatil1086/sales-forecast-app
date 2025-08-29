import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sales Forecast & Production Suggestion", page_icon="📈", layout="centered")
st.title("📈 Sales Forecast & 🏭 Production Recommendation Tool")
st.caption("Upload your CSV → forecast next 2 months → get production suggestion.")

# Required columns
REQ_COLS = {"product", "last_month_sales", "this_month_sales", "last_month_production"}

# --- CSV Upload ---
uploaded_file = st.file_uploader("📤 Upload CSV (must include product & the 3 numeric columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    # Validate columns
    if not REQ_COLS.issubset(df.columns):
        st.error("❌ CSV must have: product, last_month_sales, this_month_sales, last_month_production")
        st.stop()

    # Convert numeric cols & drop invalid rows
    for col in ["last_month_sales", "this_month_sales", "last_month_production"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["last_month_sales", "this_month_sales", "last_month_production"])
    if df.empty:
        st.error("❌ No valid numeric rows found after cleaning.")
        st.stop()

    # Add current month feature
    df["month"] = datetime.now().month

    # Train model
    X = df[["last_month_sales", "month"]]
    y = df["this_month_sales"]
    model = LinearRegression().fit(X, y)

    # Product selector
    product_list = df["product"].astype(str).unique().tolist()
    selected_product = st.selectbox("🔍 Select Product", product_list)
    prow = df[df["product"].astype(str) == selected_product].tail(1).iloc[0]

    # Extract values
    last_sales = prow["last_month_sales"]
    this_sales = prow["this_month_sales"]
    last_prod  = prow["last_month_production"]

    # Forecast next 2 months
    current_month = datetime.now().month
    month1_pred = model.predict([[this_sales, current_month]])[0]
    month2_pred = model.predict([[month1_pred, current_month]])[0]

    # Suggested production (with buffer)
    safety_factor = 1.1
    suggested_prod = round(month1_pred * safety_factor)

    # % change from last production
    if last_prod == 0:
        perc_change = 100
    else:
        perc_change = ((suggested_prod - last_prod) / last_prod) * 100

    # Recommendation text
    if perc_change > 5:
        reco = f"🔼 Increase Production by {perc_change:.1f}%"
    elif perc_change < -5:
        reco = f"🔽 Decrease Production by {abs(perc_change):.1f}%"
    else:
        reco = f"⚖️ Maintain Production (~{perc_change:.1f}%)"

    # Display results
    st.markdown(f"""
    ### 🧾 Product: `{selected_product}`

    - 🧮 **Last Month Sales**: `{last_sales:.0f}`
    - 📦 **This Month Sales**: `{this_sales:.0f}`
    - 🏗️ **Last Month Production**: `{last_prod:.0f}`

    ### 🔮 Predicted Sales
    - 📅 **Next Month**: `{month1_pred:.2f}`
    - 📅 **Month After**: `{month2_pred:.2f}`

    ### ✅ Recommendation
    {reco}  
    👉 **Suggested Next Production**: `{suggested_prod}` units
    """)

else:
    st.info("📥 Please upload a CSV file to start.")
