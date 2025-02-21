import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import requests

# MongoDB Connection Setup
@st.cache_resource
def get_mongo_client():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client

client = get_mongo_client()
db = client["weatherDataAnalysis"]  # Replace with your database name
collection = db["weatherData"]  # Replace with your collection name

# Function to fetch all data or paginated data
def fetch_data(location, limit=None):
    st.write(f"Fetching data for location_id: {location}")
    try:
        data_cursor = collection.find({'location_id': location}).limit(limit if limit else 0)
        data_list = list(data_cursor)
        
        if not data_list:
            st.warning("No data found for the given location_id.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], format="%d-%m-%Y")
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


# Streamlit Application
def main():
    st.title("Weather Data Dashboard")

    # Sidebar Input
    location_id = st.sidebar.number_input("Enter Location ID", value=0, step=1, min_value=0)
    limit = st.sidebar.number_input("Limit Number of Records (0 for all)", value=1000, step=100, min_value=0)

    # Fetch Data Button
    if st.sidebar.button("Fetch Data"):
        df = fetch_data(location_id, limit=limit if limit > 0 else None)

        if not df.empty:
            st.success(f"Fetched {len(df)} records for location_id {location_id}.")
            
            # Display raw data
            st.subheader("Raw Data")
            st.write(df)

            # Plotting options
            st.subheader("Visualizations")

            fig = px.line(
                df,
                x="time",
                y=["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
                title="Temperature Trends",
                labels={"value": "Temperature (°C)", "time": "Date"}
            )
            st.plotly_chart(fig)

            fig = px.bar(
                df,
                x="time",
                y=["rain_sum (mm)", "snowfall_sum (cm)"],
                barmode="group",
                title="Rainfall & Snowfall",
                labels={"value": "Amount", "time": "Date"}
            )
            st.plotly_chart(fig)

            fig = px.scatter(
                df,
                x="time",
                y="precipitation_hours (h)",
                title="Precipitation Hours",
                labels={"precipitation_hours (h)": "Precipitation Hours", "time": "Date"}
            )
            st.plotly_chart(fig)

            fig = px.histogram(
                df,
                x="temperature_2m_mean",
                nbins=20,
                title="Temperature Distribution",
                labels={"temperature_2m_mean": "Temperature (°C)"}
            )
            st.plotly_chart(fig)

            fig = px.box(
                df,
                x="time",
                y="precipitation_hours (h)",
                title="Precipitation Hours by Season",
                labels={"precipitation_hours (h)": "Precipitation Hours", "season": "Season"}
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()