import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset with missing value handling
def load_data():
    file_path = "Price_Agriculture_commodities_Week.csv"
    df = pd.read_csv(file_path)
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%m-%Y')
    
    # Handling missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill for time-series consistency
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Mean imputation for numeric columns
    df.fillna(df.mode().iloc[0], inplace=True)  # Mode imputation for categorical columns
    
    return df

df = load_data()

# Streamlit App
st.title("Agricultural Commodity Price Dashboard")

# Sidebar for user inputs
selected_state = st.sidebar.selectbox("Select State", df['State'].unique())
selected_commodity = st.sidebar.selectbox("Select Commodity", df['Commodity'].unique())

# Filtered data
df_filtered = df[(df['State'] == selected_state) & (df['Commodity'] == selected_commodity)]

# Price Trend Visualization
st.subheader("Price Trends Over Time")
fig = px.line(df_filtered, x='Arrival_Date', y=['Min Price', 'Max Price', 'Modal Price'], 
              labels={'value': 'Price (INR)', 'Arrival_Date': 'Date'}, title=f"{selected_commodity} Price Trends in {selected_state}")
st.plotly_chart(fig)

# Market Comparison
st.subheader("Market Price Comparison")
df_market = df_filtered.groupby('Market').mean().reset_index()
fig2 = px.bar(df_market, x='Market', y='Modal Price', 
              labels={'Modal Price': 'Average Modal Price (INR)', 'Market': 'Market'}, 
              title=f"Market-wise Price Comparison for {selected_commodity}")
st.plotly_chart(fig2)

# Price Forecasting
st.subheader("Price Forecasting")

# Preprocessing data for ML
le_state = LabelEncoder()
df['State_Code'] = le_state.fit_transform(df['State'])
le_commodity = LabelEncoder()
df['Commodity_Code'] = le_commodity.fit_transform(df['Commodity'])

df['Days'] = (df['Arrival_Date'] - df['Arrival_Date'].min()).dt.days

features = ['State_Code', 'Commodity_Code', 'Days']
target = 'Modal Price'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_price(state, commodity, future_days):
    state_code = le_state.transform([state])[0]
    commodity_code = le_commodity.transform([commodity])[0]
    future_date = df['Days'].max() + future_days
    return model.predict([[state_code, commodity_code, future_date]])[0]

future_days = st.slider("Select days into the future for prediction", 1, 30, 7)
predicted_price = predict_price(selected_state, selected_commodity, future_days)
st.write(f"Predicted Modal Price for {selected_commodity} in {selected_state} after {future_days} days: ₹{predicted_price:.2f}")
