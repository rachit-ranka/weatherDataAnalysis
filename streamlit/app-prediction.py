import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

# Function to fetch latitude and longitude for a given location


def get_lat_lon(location_name):
    api_key = "7ea4c213fda44209b4685e71ab062176"  # Replace with your OpenCage API key
    api_url = f"https://api.opencagedata.com/geocode/v1/json?q={location_name}&key={api_key}"
    
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            latitude = data['results'][0]['geometry']['lat']
            longitude = data['results'][0]['geometry']['lng']
            return latitude, longitude
        else:
            raise ValueError("No results found for the given location.")
    else:
        raise Exception(f"Geocoding API request failed with status code: {response.status_code}")


# Function to fetch data from the provided API and calculate daily average temperature
def fetch_past_30_days_avg_temp(lat, lon, start_date, end_date):
    print(lat, lon, start_date, end_date)
    api_url = (
        f"https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min"
        f"&timezone=auto"
    )
    response = requests.get(api_url)
    print(response)
    if response.status_code == 200:
        data = response.json()
        # Extract daily min and max temperatures
        daily_min_temps = data['daily']['temperature_2m_min']
        daily_max_temps = data['daily']['temperature_2m_max']
        
        # Calculate daily average temperatures
        daily_avg_temps = [(min_temp + max_temp) / 2 for min_temp, max_temp in zip(daily_min_temps, daily_max_temps)]
        
        if len(daily_avg_temps) < 30:
            raise ValueError("Insufficient data! API must return at least 30 days of data.")
        
        return daily_avg_temps[-30:]  # Get the last 30 days
    else:
        raise Exception(f"API request failed with status code: {response.status_code}")

# Load the saved model, scaler, and configuration
try:
    loaded_model = load_model("Forecasting_model/temperature_forecast_model.h5")
    scaler = joblib.load("scaler.pkl")
    config = joblib.load("config.pkl")
    prediction_days = config['prediction_days']
except Exception as e:
    st.error(f"Error loading model or files: {e}")
    st.stop()

# Streamlit UI
st.title("Temperature Prediction Dashboard")

# User input for location
location_name = st.text_input("Enter the location for prediction:", "Bangalore")

# Predict button
if st.button("Predict Temperature"):
    try:
        print("Getting Latitudes")
        # Get latitude and longitude for the location
        lat, lon = get_lat_lon(location_name)
        # print(lat,lon)
        # st.success(f"Location found: Latitude = {lat}, Longitude = {lon}")

        # Calculate start and end dates
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=31)).strftime('%Y-%m-%d')

        # Fetch past 30 days of average temperature data
        past_30_days_avg_temp = fetch_past_30_days_avg_temp(lat, lon, start_date, end_date)
        # st.success("Fetched data from API successfully!")

        # Scale the average temperatures
        last_30_days_scaled = scaler.transform(np.array(past_30_days_avg_temp).reshape(-1, 1))

        # Reshape the input to match the model's expected input shape
        input_data = np.reshape(last_30_days_scaled, (1, prediction_days, 1))  # (samples, time_steps, features)

        # Predict the next day's temperature
        future_temp_scaled = loaded_model.predict(input_data)

        # Inverse transform the prediction to get the original scale
        future_temp = scaler.inverse_transform(future_temp_scaled)
        st.success(f"Predicted Temperature: {future_temp[0][0]:.2f} °C")
    except Exception as e:
        st.error(f"Error: {e}")