import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Population data: 2011 Census (official), 2024 & 2025 estimates from Technical Group / StatisticsTimes
# (populations in individuals)
data = {
    'Year': [2011, 2024, 2025],
    'Uttar Pradesh': [199812341, 236484000, 241300000],
    'Bihar': [104099452, 129205000, 131000000],
    'Maharashtra': [112374333, 126710000, None],
    'West Bengal': [91276115, 99243000, None],
    'Madhya Pradesh': [72626809, None, None],
    'Tamil Nadu': [72147030, 76936000, 77165000],
    'Rajasthan': [68548437, None, 83060000],
    'Karnataka': [61095297, None, None],
    'Gujarat': [60439692, None, 73513000],
    'Andhra Pradesh': [84580777, None, None],
    'Odisha': [41974218, None, None],
    'Telangana': [None, 38317000, 38499000],
    'Kerala': [33406061, None, 36111000],
    'Jharkhand': [32988134, None, 40626000],
    'Assam': [31205576, None, 36493000],
    # ... additional states/UTs can be added similarly
    'Delhi': [16787941, 21490000, 22277000],
    'Punjab': [27743338, None, 31188000],
    'Haryana': [25351462, None, 31057000],
    'Chhattisgarh': [25545198, None, 30982000],
    'Uttarakhand': [10086292, None, 11913000],
    'Goa': [1458545, None, 1593000],
    'Tripura': [3673917, None, 4232000],
    'Meghalaya': [2966889, None, 3417000],
    'Manipur': [2570390, None, 3289000],
    'Nagaland': [1978502, None, 2258000],
    'Mizoram': [1097206, None, 1264000],
    'Sikkim': [610577, None, 703000],
    'Andaman & Nicobar': [380581, None, 405000],
    'Chandigarh': [1055450, None, 1257000],
    'Puducherry': [1247953, None, 1732000],
    'Lakshadweep': [64473, None, 69000],
}

# Create DataFrame
df = pd.DataFrame(data).set_index('Year')

# Plot population trends
plt.figure(figsize=(16, 12))
for state in df.columns:
    plt.plot(df.index, df[state], marker='o', label=state)

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Trends of Indian States & UTs (2011, 2024, 2025)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Forecasting function using linear regression
def forecast_population(state_name, start_year=2025, end_year=2030):
    y = df[state_name].dropna()
    x = y.index.values.reshape(-1, 1)
    model = LinearRegression().fit(x, y.values)
    future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
    preds = model.predict(future_years)
    return list(zip(future_years.flatten(), preds.astype(int)))

# Example: Forecasting for Uttar Pradesh
state_to_forecast = 'Uttar Pradesh'
forecast = forecast_population(state_to_forecast)

print(f"\nForecasted population for {state_to_forecast} (2026-2030):")
for year, pop in forecast:
    print(f"{year}: {pop:,}")

