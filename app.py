import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime

st.set_page_config(page_title="Sales Forecast & Production Suggestion", page_icon="📈", layout="centered")
st.title("📈 Sales Forecast & 🏭 Production Recommendation Tool")
st.caption("Upload your CSV → train model → forecast next 2 months → get production suggestion.")

req_cols = {"product", "last_month_sales", "this_month_sales", "last_month_production"}

# --- CSV Upload ---
uploaded_file = st.file_uploader("📤 Upload CSV (must include product & the 3 numeric columns)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    # Validate columns
    if not req_cols.issubset(df.columns):
        st.error("❌ CSV must have columns: 'product', 'last_month_sales', 'this_month_sales', 'last_month_production'")
        st.stop()

    # Ensure numeric types
    for col in ["last_month_sales", "this_month_sales", "last_month_production"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean = df.dropna(subset=["last_month_sales", "this_month_sales", "last_month_production"]).copy()
    if clean.empty:
        st.error("❌ After cleaning, no valid numeric rows remain.")
        st.stop()

    # --- Add seasonality / month feature ---
    current_month = datetime.now().month
    clean["month"] = current_month

    # Train Linear Regression with month as additional feature
    X = clean[["last_month_sales", "month"]].values
    y = clean["this_month_sales"].values
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    st.success("✅ Model trained successfully with seasonality feature.")

    # Product selector
    product_list = clean["product"].astype(str).unique().tolist()
    selected_product = st.selectbox("🔍 Select Product", product_list)
    prow = clean[clean["product"].astype(str) == selected_product].tail(1).iloc[0]

    last_sales = float(prow["last_month_sales"])
    this_sales = float(prow["this_month_sales"])
    last_prod  = float(prow["last_month_production"])

    # Predict next 2 months autoregressively
    month1_pred = float(model.predict(np.array([[this_sales, current_month]]))[0])
    month2_pred = float(model.predict(np.array([[month1_pred, current_month]]))[0])

    # --- Safety buffer & suggested production ---
    safety_factor = 1.1  # 10% buffer
    suggested_next_production = max(0, round(month1_pred * safety_factor))

    # --- % change recommendation ---
    if last_prod == 0:
        perc_change = 100
    else:
        perc_change = ((suggested_next_production - last_prod) / last_prod) * 100

    if perc_change > 5:
        reco = f"🔼 Increase Production by {perc_change:.1f}%"
    elif perc_change < -5:
        reco = f"🔽 Decrease Production by {abs(perc_change):.1f}%"
    else:
        reco = f"⚖️ Maintain Production (~{perc_change:.1f}%)"

    # Display results
    st.markdown(f'''
    ### 🧾 Product: `{selected_product}`

    - 🧮 **Last Month Sales**: `{last_sales:.0f}`
    - 📦 **This Month Sales**: `{this_sales:.0f}`
    - 🏗️ **Last Month Production**: `{last_prod:.0f}`

    ### 🔮 Predicted Sales
    - 📅 **Next Month**: `{month1_pred:.2f}`
    - 📅 **Month After**: `{month2_pred:.2f}`

    ### ✅ Recommendation
    {reco}  
    👉 **Suggested Next Production**: `{suggested_next_production}` units
    ''')

else:
    st.info("Waiting for CSV upload.")
