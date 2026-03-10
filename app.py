import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Saudi House Rent Dashboard", layout="wide")

# Load data
df = pd.read_csv("cleaned_house_data.csv")
model = joblib.load("house_model.pkl")

# Title
st.title("Saudi House Rent Dashboard & Prediction")

# Sidebar
st.sidebar.header("Dataset Info")
st.sidebar.write("""
This dataset contains house rental information in Saudi Arabia.
The model predicts **annual rent price** based on property features.
""")

# Filters
st.sidebar.header("Filters")

bedrooms_filter = st.sidebar.multiselect(
    "Bedrooms",
    options=sorted(df["bedrooms"].unique()),
    default=sorted(df["bedrooms"].unique())
)

price_range = st.sidebar.slider(
    "Price Range",
    int(df["price"].min()),
    int(df["price"].max()),
    (int(df["price"].min()), int(df["price"].max()))
)

filtered_df = df[
    (df["bedrooms"].isin(bedrooms_filter)) &
    (df["price"].between(price_range[0], price_range[1]))
]

# Metrics
st.subheader("Market Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Properties", len(filtered_df))
col2.metric("Average Rent", f"{filtered_df['price'].mean():,.0f} SAR")
col3.metric("Highest Rent", f"{filtered_df['price'].max():,.0f} SAR")

# Data preview
st.subheader("Sample Data")
st.dataframe(filtered_df.head())

# Visualizations
st.subheader("Market Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["price"], kde=True, ax=ax)
    ax.set_title("Rent Price Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="size", y="price", ax=ax)
    ax.set_title("Property Size vs Rent Price")
    st.pyplot(fig)

# Insights
st.subheader("Insights")

st.write("""
- Larger houses tend to have higher rental prices.
- Houses with more bedrooms usually cost more.
- Property size is one of the strongest factors affecting rent.
""")

# Prediction section
st.subheader("Predict Rental Price")

col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Size (m²)", min_value=50, max_value=1000, value=300)
    property_age = st.number_input("Property Age", min_value=0, max_value=50, value=5)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=3)
    livingrooms = st.number_input("Living Rooms", min_value=1, max_value=5, value=2)

with col2:
    kitchen = st.selectbox("Kitchen", [0,1])
    garage = st.selectbox("Garage", [0,1])
    driver_room = st.selectbox("Driver Room", [0,1])
    maid_room = st.selectbox("Maid Room", [0,1])
    furnished = st.selectbox("Furnished", [0,1])

col3, col4 = st.columns(2)

with col3:
    ac = st.selectbox("AC", [0,1])
    roof = st.selectbox("Roof", [0,1])
    pool = st.selectbox("Pool", [0,1])
    frontyard = st.selectbox("Front Yard", [0,1])

with col4:
    basement = st.selectbox("Basement", [0,1])
    duplex = st.selectbox("Duplex", [0,1])
    stairs = st.selectbox("Stairs", [0,1])
    elevator = st.selectbox("Elevator", [0,1])
    fireplace = st.selectbox("Fireplace", [0,1])

city = st.selectbox("City", ["الرياض","جدة","الدمام","الخبر"])

if st.button("Predict Price"):

    input_data = pd.DataFrame(columns=df.drop("price", axis=1).columns)
    input_data.loc[0] = 0

    input_data["size"] = size
    input_data["property_age"] = property_age
    input_data["bedrooms"] = bedrooms
    input_data["bathrooms"] = bathrooms
    input_data["livingrooms"] = livingrooms
    input_data["kitchen"] = kitchen
    input_data["garage"] = garage
    input_data["driver_room"] = driver_room
    input_data["maid_room"] = maid_room
    input_data["furnished"] = furnished
    input_data["ac"] = ac
    input_data["roof"] = roof
    input_data["pool"] = pool
    input_data["frontyard"] = frontyard
    input_data["basement"] = basement
    input_data["duplex"] = duplex
    input_data["stairs"] = stairs
    input_data["elevator"] = elevator
    input_data["fireplace"] = fireplace

    city_column = f"city_{city}"

    if city_column in input_data.columns:
        input_data[city_column] = 1

    prediction = model.predict(input_data)

    st.success(f"Predicted Annual Rent: {prediction[0]:,.0f} SAR")
