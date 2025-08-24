import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

st.set_page_config(page_title="Sales Forecast & Production Suggestion")

st.title("📈 Sales Forecast & 🏭 Production Recommendation Tool")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV (with sales & production data)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data", df)

    if {"last_month_sales", "this_month_sales"}.issubset(df.columns):
        # Train model
        X = df[["last_month_sales"]]
        y = df["this_month_sales"]
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        st.success("✅ Model trained successfully!")

        # Product selection
        product_list = df["product"].unique()
        selected_product = st.selectbox("🔍 Select Product", product_list)

        row = df[df["product"] == selected_product].iloc[-1]
        last_sales = row["last_month_sales"]
        this_sales = row["this_month_sales"]
        last_prod = row["last_month_production"]

        # Load and predict
        model = joblib.load("model.pkl")
        month1_pred = model.predict([[this_sales]])[0]
        month2_pred = model.predict([[month1_pred]])[0]

        # Recommendation
        if this_sales > last_sales:
            reco = "🔼 Increase Production"
        elif this_sales < last_sales:
            reco = "🔽 Decrease Production"
        else:
            reco = "⚖️ Maintain Production"

        st.markdown(f"""
        ### 🧾 Product: `{selected_product}`

        - 🧮 **Last Month Sales**: {last_sales}
        - 📦 **This Month Sales**: {this_sales}
        - 🏗️ **Last Month Production**: {last_prod}

        ### 🔮 Predicted Sales:
        - 📅 Next Month: **{month1_pred:.2f}**
        - 📅 Month After: **{month2_pred:.2f}**

        ### ✅ Recommendation: {reco}
        """)
    else:
        st.error("❌ CSV must have 'last_month_sales', 'this_month_sales', 'last_month_production'")