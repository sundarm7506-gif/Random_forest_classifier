import streamlit as st
import pandas as pd
import joblib

# Load model & fruit encoder
model = joblib.load("rf_model.pkl")
fruit_encoder = joblib.load("fruit_encoder.pkl")

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎")

st.title("🍎 Fruit Classification App")
st.write("Predict fruit type using Random Forest Classifier")

# Inputs
size = st.number_input("Size (cm)", min_value=0.0)
shape = st.selectbox("Shape", ["round", "oval", "long"])
weight = st.number_input("Weight (g)", min_value=0.0)
price = st.number_input("Average Price (₹)", min_value=0.0)

# IMPORTANT: manual encoding (matches training)
shape_map = {
    "round": 0,
    "oval": 1,
    "long": 2
}

if st.button("Predict Fruit"):
    shape_encoded = shape_map[shape]

    input_df = pd.DataFrame(
        [[size, shape_encoded, weight, price]],
        columns=["size (cm)", "shape", "weight (g)", "avg_price (₹)"]
    )

    prediction = model.predict(input_df)
    fruit = fruit_encoder.inverse_transform(prediction)

    st.success(f"🍓 Predicted Fruit: **{fruit[0]}**")

        