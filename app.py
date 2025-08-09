import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Indian Cities Population", layout="centered")
st.title("🏙 Population of Major Indian Cities in India")

# Population data of major cities (You can expand this list)
data = {
    "City": [
        "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad",
        "Chennai", "Kolkata", "Pune", "Jaipur", "Surat"
    ],
    "Population (Millions)": [20.4, 18.6, 12.3, 10.5, 8.4, 8.2, 14.8, 6.6, 4.0, 6.2]
}

# Create DataFrame
df = pd.DataFrame(data)

# Show table
st.subheader("📊 Indian Cities Population Data")
st.dataframe(df)

# Graph type selection
chart_type = st.selectbox("Choose Graph Type:", ["Bar Chart", "Line Chart", "Pie Chart"])

# Plot graph
plt.figure(figsize=(10, 6))
if chart_type == "Bar Chart":
    plt.bar(df["City"], df["Population (Millions)"], color='skyblue')
    plt.ylabel("Population (Millions)")
    plt.title("Population of Major Indian Cities")
elif chart_type == "Line Chart":
    plt.plot(df["City"], df["Population (Millions)"], marker='o', color='green')
    plt.ylabel("Population (Millions)")
    plt.title("Population of Major Indian Cities")
elif chart_type == "Pie Chart":
    plt.pie(df["Population (Millions)"], labels=df["City"], autopct='%1.1f%%', startangle=140)
    plt.title("Population Share of Major Indian Cities")

st.pyplot(plt)
