import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Saudi House Rent Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-title {
    color:#0F4C75;
    font-size:42px;
    font-weight:700;
}
.section-title {
    color:#3282B8;
    font-size:26px;
    font-weight:600;
    margin-top:20px;
}
.prediction {
    color:#27AE60;
    font-size:32px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# Load data and model
df = pd.read_csv("cleaned_house_data.csv")
model = joblib.load("house_model.pkl")

# Title
st.markdown('<p class="main-title">Saudi House Rent Dashboard & Prediction</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Dataset Description")
st.sidebar.write("""
This dataset contains house rental information in Saudi cities.

Features include:
- Property size
- Bedrooms
- Bathrooms
- Living rooms
- Property location
- Rental price
""")

# Filters
st.sidebar.header("Filters")

bedroom_filter = st.sidebar.multiselect(
    "Bedrooms",
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

# Metrics
st.markdown('<p class="section-title">Market Overview</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("Total Properties", len(filtered_df))
col2.metric("Average Rent", f"{filtered_df['price'].mean():,.0f} SAR")
col3.metric("Highest Rent", f"{filtered_df['price'].max():,.0f} SAR")

# Data preview
st.markdown('<p class="section-title">Sample Data</p>', unsafe_allow_html=True)
st.dataframe(filtered_df.head())

# Visualizations
st.markdown('<p class="section-title">Market Visualizations</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df["price"], kde=True, color="#0F4C75", ax=ax1)
    ax1.set_title("Rent Price Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="size", y="price", data=filtered_df, color="#1FAB89", ax=ax2)
    ax2.set_title("Property Size vs Rent Price")
    st.pyplot(fig2)

# Insights
st.markdown('<p class="section-title">Market Insights</p>', unsafe_allow_html=True)

st.write("""
• Larger properties generally command higher rental prices.  

• Houses with more bedrooms and bathrooms tend to have higher market value.  

• Property size is the strongest factor influencing rent price.  

• Rental prices vary significantly between different Saudi cities depending on demand and location.
""")

# Prediction
st.markdown('<p class="section-title">Predict Rental Price</p>', unsafe_allow_html=True)

size = st.number_input("Property Size", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
livingrooms = st.number_input("Living Rooms", min_value=0)

city = st.selectbox("City", ["الرياض", "جدة", "الدمام", "الخبر"])

if st.button("Predict Rent Price"):

    input_data = pd.DataFrame(columns=df.drop("price", axis=1).columns)
    input_data.loc[0] = 0

    input_data["size"] = size
    input_data["bedrooms"] = bedrooms
    input_data["bathrooms"] = bathrooms
    input_data["livingrooms"] = livingrooms

    city_column = f"city_{city}"
    if city_column in input_data.columns:
        input_data[city_column] = 1

    prediction = model.predict(input_data)

    st.markdown(
        f"<p class='prediction'>Predicted Rent: {prediction[0]:,.0f} SAR</p>",
        unsafe_allow_html=True
    )
