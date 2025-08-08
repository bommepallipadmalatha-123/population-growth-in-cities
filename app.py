# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="State Population Growth", layout="centered")
st.title("📈 Population Growth Prediction in Indian States")

# All Indian States list (as of 2025)
states = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim',
    'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand',
    'West Bengal'
]

# Sample historical data (you should replace this with real data or load from CSV)
@st.cache_data
def load_data():
    # Dummy population data (use actual population data in real project)
    data = []
    for state in states:
        for year in [2000, 2005, 2010, 2015, 2020]:
            population = np.random.randint(5_000_000, 120_000_000)  # Dummy values
            data.append([state, year, population])
    df = pd.DataFrame(data, columns=["State", "Year", "Population"])
    return df

df = load_data()

# Sidebar: state and future year selection
st.sidebar.header("Choose Input")
selected_state = st.sidebar.selectbox("Select a State", df["State"].unique())
selected_year = st.sidebar.slider("Select Future Year to Predict", 2025, 2050, 2030)

# Filter for selected state
state_data = df[df["State"] == selected_state]
X = state_data["Year"].values.reshape(-1, 1)
y = state_data["Population"].values.reshape(-1, 1)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict for selected year
predicted_population = model.predict([[selected_year]])[0][0]

# Display result
st.subheader(f"📍 {selected_state} - Predicted Population in {selected_year}")
st.success(f"Estimated population: **{int(predicted_population):,}** people")

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Historical')
ax.plot(X, model.predict(X), color='green', label='Trend')
ax.scatter(selected_year, predicted_population, color='red', label='Prediction')
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.set_title(f"Population Growth: {selected_state}")
ax.legend()
st.pyplot(fig)
