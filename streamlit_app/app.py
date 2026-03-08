import streamlit as st
import joblib
import pandas as pd

model = joblib.load("house_model.pkl")

df = pd.read_csv("cleaned_house_data.csv")

st.title("House Price Prediction")

size = st.number_input("Property Size")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
livingrooms = st.number_input("Living Rooms")

if st.button("Predict Price"):

    data = pd.DataFrame(columns=df.drop("price", axis=1).columns)

    data.loc[0, "size"] = size
    data.loc[0, "bedrooms"] = bedrooms
    data.loc[0, "bathrooms"] = bathrooms
    data.loc[0, "livingrooms"] = livingrooms

    data = data.fillna(0)

    prediction = model.predict(data)

    st.success(f"Predicted Price: {prediction[0]}")