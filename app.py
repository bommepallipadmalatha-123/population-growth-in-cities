import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="City Population Growth", layout="centered")
st.title("📈 Population Growth Prediction in Indian Cities")

# List of major Indian cities
cities = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Surat", "Pune", "Jaipur",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara"
]

# Load or generate dummy data
@st.cache_data
def load_data():
    data = []
    for city in cities:
        for year in [2000, 2005, 2010, 2015, 2020]:
            population = np.random.randint(1_000_000, 30_000_000)  # Smaller than states
            data.append([city, year, population])
    df = pd.DataFrame(data, columns=["City", "Year", "Population"])
    return df

df = load_data()

# Sidebar inputs
st.sidebar.header("Choose Input")
selected_city = st.sidebar.selectbox("Select a City", df["City"].unique())
selected_year = st.sidebar.slider("Select Future Year to Predict", 2025, 2050, 2030)

# Filter data for the selected city
city_data = df[df["City"] == selected_city]

# Prepare features and target
X = city_data["Year"].values.reshape(-1, 1)
y = city_data["Population"].values

@st.cache_data
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(X, y)

# Predict population for selected future year
predicted_population = model.predict([[selected_year]])[0]
predicted_population = max(predicted_population, 0)

# Display prediction
st.subheader(f"📍 {selected_city} - Predicted Population in {selected_year}")
st.success(f"Estimated population: *{int(predicted_population):,}* people")

# Plotting
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X, y, color='blue', label='Historical Data')
ax.plot(X, model.predict(X.reshape(-1,1)), color='green', label='Trend Line')
ax.scatter(selected_year, predicted_population, color='red', label='Prediction')
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.set_title(f"Population Growth: {selected_city}")
ax.legend()
ax.grid(True)
plt.xticks(list(X.flatten()) + [selected_year])
ax.ticklabel_format(axis='y', style='plain')
st.pyplot(fig)

# Note about dummy data
st.markdown(
    "<small><i>Note: Population data shown is randomly generated dummy data for demo purposes only.</i></small>",
    unsafe_allow_html=True
)
