# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("🌆 Population Growth Prediction in Cities")
st.write("This app predicts future population growth for Indian cities using Linear Regression.")

# Load dataset
@st.cache_data
def load_data():
    # Sample data - you should replace with a full dataset of Indian cities
    data = {
        'City': ['Mumbai'] * 5 + ['Delhi'] * 5 + ['Bangalore'] * 5,
        'Year': [2000, 2005, 2010, 2015, 2020] * 3,
        'Population': [11978450, 12442373, 12990000, 13500000, 14000000,
                       11034555, 11500000, 12000000, 12500000, 13000000,
                       6500000, 7000000, 7500000, 8000000, 8500000]
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# Sidebar input
city = st.sidebar.selectbox("Select a city", df['City'].unique())
future_year = st.sidebar.slider("Select future year to predict", 2025, 2050, 2030)

# Filter data
city_data = df[df['City'] == city]
X = city_data['Year'].values.reshape(-1, 1)
y = city_data['Population'].values.reshape(-1, 1)

# Model
model = LinearRegression()
model.fit(X, y)
future_pred = model.predict(np.array([[future_year]]))[0][0]

# Display prediction
st.subheader(f"📈 Predicted population of {city} in {future_year}:")
st.success(f"**{int(future_pred):,}** people")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label="Historical Data")
plt.plot(X, model.predict(X), color='green', label="Regression Line")
plt.scatter(future_year, future_pred, color='red', label="Prediction")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title(f"Population Growth Prediction for {city}")
plt.legend()
st.pyplot(plt)
