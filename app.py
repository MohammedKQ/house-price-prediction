import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_house_data.csv")
model = joblib.load("house_model.pkl")

st.title("House Price Dashboard & Prediction")

st.sidebar.header("Dataset Description")

st.sidebar.write("""
This dataset contains house information such as:
size, bedrooms, bathrooms, livingrooms and price.
""")

st.sidebar.header("Filters")

bedroom_filter = st.sidebar.multiselect(
    "Select Bedrooms",
    df["bedrooms"].unique(),
    default=df["bedrooms"].unique()
)

price_range = st.sidebar.slider(
    "Price Range",
    int(df["price"].min()),
    int(df["price"].max()),
    (int(df["price"].min()), int(df["price"].max()))
)

filtered_df = df[
    (df["bedrooms"].isin(bedroom_filter)) &
    (df["price"].between(price_range[0], price_range[1]))
]

st.header("Data Preview")
st.dataframe(filtered_df.head())

st.header("Summary Statistics")
st.write(filtered_df.describe())

st.header("Visualizations")

fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["price"], kde=True, ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.scatterplot(x="size", y="price", data=filtered_df, ax=ax2)
st.pyplot(fig2)

st.header("Insights")

st.write("""
- Larger houses usually have higher prices.
- Bedrooms influence house value.
- House size strongly affects price.
""")

st.header("Predict House Price")

size = st.number_input("Property Size")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
livingrooms = st.number_input("Living Rooms")

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "size":[size],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms],
        "livingrooms":[livingrooms]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: {prediction[0]}")