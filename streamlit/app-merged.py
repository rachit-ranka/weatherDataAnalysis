import streamlit as st

# App title
st.title("Weather Data Analysis App")
 
# Create tabs
tab1, tab2 ,tab3 = st.tabs(["Real Time Weather Data Analysis", "Historical Weather Data Analysis", "Weather Prediction"])

# Content for Tab 1
with tab1:
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("WeatherDataAnalysis") \
        .getOrCreate()

    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    import requests
    import pandas as pd

    API_KEY = "5fee2ce62aa847d6bea52d442e42814a"
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    # Define schema
    schema = StructType([
        StructField("City", StringType(), True),
        StructField("Temperature", DoubleType(), True),
        StructField("Humidity", DoubleType(), True),
        StructField("WindSpeed", DoubleType(), True),
        StructField("Weather", StringType(), True),
        StructField("Latitude", DoubleType(), True),
        StructField("Longitude", DoubleType(), True)
    ])

    # Function to fetch weather data
    def fetch_weather_data(city):
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(base_url, params=params)
        data = response.json()
        if data.get("main"):
            return Row(
                City=data["name"],
                Temperature=float(data["main"]["temp"]),
                Humidity=float(data["main"]["humidity"]),
                WindSpeed=float(data["wind"]["speed"]),
                Weather=data["weather"][0]["description"],
                Latitude=float(data["coord"]["lat"]),
                Longitude=float(data["coord"]["lon"])
            )

    # Cities list
    cities = [
        "Srinagar", "Leh", "Shimla", "Chandigarh", "Dehradun", "New Delhi", "Jaipur", "Amritsar",
        "Ahmedabad", "Mumbai", "Jaisalmer", "Panaji",
        "Bhopal", "Raipur", "Nagpur",
        "Kolkata", "Patna", "Darjeeling",
        "Bengaluru", "Chennai", "Hyderabad", "Thiruvananthapuram",
        "Guwahati", "Shillong", "Aizawl"
    ]

    # Fetch data
    weather_rows = [fetch_weather_data(city) for city in cities if fetch_weather_data(city) is not None]

    # Create DataFrame with defined schema
    weather_df = spark.createDataFrame(weather_rows, schema=schema)

    # Show the DataFrame
    weather_df.show()
    pandas_df = weather_df.toPandas()

    # Add a severity column based on temperature
    from pyspark.sql.functions import when

    weather_df = weather_df.withColumn(
        "Severity",
        when(weather_df.Temperature > 35, "High")
        .when(weather_df.Temperature > 20, "Moderate")
        .otherwise("Low")
    )

    # Group by Severity
    severity_count = weather_df.groupBy("Severity").count()
    severity_count.show()

    # Calculate the average temperature, humidity, and wind speed
    average_temp = pandas_df["Temperature"].mean()
    average_humidity = pandas_df["Humidity"].mean()
    average_wind_speed = pandas_df["WindSpeed"].mean()


    #######
    st.title("Real Time Weather Data Analysis")

    st.subheader("Weather Gauge Charts")

    def create_gauge_chart(value, title, min_val, max_val):
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Create a circular gauge
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Define the color palette with a gradient-like effect
        colors = ["lightblue", "white"]
        
        # Compute the angle based on the value
        angle = (value - min_val) / (max_val - min_val) * 180

        # Create the gauge segments
        ax.pie([angle, 180 - angle], startangle=0, colors=colors, wedgeprops={'width': 0.3})
        
        # Add a border circle to give it a more polished look
        circle = plt.Circle((0, 0), 0.7, color='black', fill=False, linewidth=3)
        ax.add_artist(circle)

        # Add title and value text
        ax.text(0, 0, f"{value:.2f}", ha='center', va='center', fontweight="bold", fontsize=20, color="black")
        ax.text(0, -0.2, title, ha='center', va='center', fontweight="bold", fontsize=14, color='gray')
        
        # Remove axes
        ax.axis('off')
        
        return fig

    # Streamlit app
    st.subheader("Average Temperature, Humidity, and Wind Speed")

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Populate columns with visualizations
    with col1:
        st.pyplot(create_gauge_chart(average_temp, "Temp (°C)", min_val=0, max_val=50))

    with col2:
        st.pyplot(create_gauge_chart(average_humidity, "Humidity (%)", min_val=0, max_val=100))

    with col3:
        st.pyplot(create_gauge_chart(average_wind_speed, "Wind Speed (m/s)", min_val=0, max_val=20))


    #############





    # Function to create a gauge chart
    def create_gauge_chart(value, title, min_val, max_val):
        if value < min_val:
            value = min_val
        elif value > max_val:
            value = max_val
        
        fig, ax = plt.subplots(figsize=(2, 2))  # Smaller figure size for better layout
        
        # Compute the angle based on the value
        angle = (value - min_val) / (max_val - min_val) * 180

        # Create the gauge segments
        ax.pie(
            [angle, 180 - angle],
            startangle=0,
            colors=["lightblue", "white"],
            wedgeprops={"width": 0.3},
        )
        
        # Add title and value text
        ax.text(0, 0, f"{value:.1f}", ha="center", va="center", fontweight="bold", fontsize=12, color="black")
        ax.text(0, -1.5, title, ha="center", va="center", fontweight="bold", fontsize=10, color="gray")
        
        # Remove axes
        ax.axis("off")
        
        return fig

    # Function to display the weather data for a specific city
    def display_city_weather_data(city_name, pandas_df):
        # Check if the city exists in the DataFrame
        if city_name in pandas_df["City"].values:
            city_data = pandas_df.loc[pandas_df["City"] == city_name].iloc[0]
            
            # Fetch the weather data for the city
            city_temp = city_data["Temperature"]
            city_humidity = city_data["Humidity"]
            city_wind_speed = city_data["WindSpeed"]
            
            st.subheader(f"Weather Data for {city_name}")

            # Create columns for displaying charts side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.pyplot(create_gauge_chart(city_temp, "Temp (°C)", min_val=0, max_val=50))
            with col2:
                st.pyplot(create_gauge_chart(city_humidity, "Humidity (%)", min_val=0, max_val=100))
            with col3:
                st.pyplot(create_gauge_chart(city_wind_speed, "Wind Speed (m/s)", min_val=0, max_val=20))
        else:
            st.error(f"City '{city_name}' not found in the dataset.")


    # Streamlit app

    # User input for city selection
    city_name = st.selectbox("Select a City", pandas_df["City"])

    # Display weather data for the selected city
    if city_name:
        display_city_weather_data(city_name, pandas_df)


    ##########


    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Define weather categories mapped to colors
    weather_colors = {
        "overcast clouds": "gray",
        "light intensity drizzle": "lightblue",
        "broken clouds": "purple",
        "fog": "black",
        "few clouds": "skyblue",
        "smoke": "orange",
        "clear sky": "blue",
        "haze": "yellow"
    }

    # Streamlit app title
    st.title("Weather Forecast in Indian Cities")

    # Create a figure with Cartopy's projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())  # Set bounds for India

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linestyle=":")

    # Plot the data points
    for index, row in pandas_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        city, weather, temp = row["City"], row["Weather"], row["Temperature"]
        ax.plot(lon, lat, 'o', color=weather_colors.get(weather, "gray"), markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon + 0.5, lat + 0.5, f"{city}\n{temp}°C", fontsize=9, transform=ccrs.PlateCarree())

    # Add a legend
    unique_weathers = pandas_df["Weather"].unique()
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=weather_colors.get(weather, "gray"),
            markersize=10,
            label=weather
        ) for weather in unique_weathers
    ]
    ax.legend(handles=handles, loc='lower left', title="Weather Conditions", fontsize=9)

    # Set the title
    plt.title("Weather Forecast in Indian Cities")

    # Display the plot in Streamlit
    st.pyplot(fig)



    #########



    import streamlit as st
    import matplotlib.pyplot as plt

    # Assuming severity_count is a PySpark DataFrame, convert it to pandas
    severity_pandas_df = severity_count.toPandas()

    # Define a color mapping for severity levels
    color_mapping = {
        "Low": "green",
        "Moderate": "yellow",
        "High": "orange",
        "Severe": "red"
    }

    # Map colors to the severity levels in the DataFrame
    severity_pandas_df["Color"] = severity_pandas_df["Severity"].map(color_mapping)

    # Streamlit App
    st.title("Severity Count Visualization")

    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        severity_pandas_df["Severity"], 
        severity_pandas_df["count"], 
        color=severity_pandas_df["Color"]
    )

    # Add title and labels
    ax.set_title("Severity Count by Region")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Count")

    # Display the chart in Streamlit
    st.pyplot(fig)


    ###########



    import streamlit as st
    import matplotlib.pyplot as plt

    # Assuming pandas_df is your DataFrame
    # Group by 'Weather' to count the number of cities for each weather type
    weather_counts = pandas_df.groupby("Weather").size()

    # Define a color mapping for weather types
    weather_colors = {
        "overcast clouds": "gray",
        "light intensity drizzle": "lightblue",
        "broken clouds": "purple",
        "fog": "white",
        "few clouds": "skyblue",
        "smoke": "orange",
        "clear sky": "blue",
        "haze": "yellow",
    }

    # Prepare data for the bar chart
    weather_counts_sorted = weather_counts.sort_values(ascending=False)
    weather_types = weather_counts_sorted.index
    counts = weather_counts_sorted.values
    colors = [
        weather_colors.get(weather, "black") 
        for weather in weather_types 
        if weather_colors.get(weather, "black") != "white"
    ]

    # Streamlit app
    st.title("Weather Distribution Across Indian Cities")

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(weather_types, counts, color=colors)

    # Add labels and title
    ax.set_title("Weather Distribution Across Indian Cities", fontsize=16)
    ax.set_xlabel("Weather Type", fontsize=12)
    ax.set_ylabel("Number of Cities", fontsize=12)
    ax.set_xticks(range(len(weather_types)))
    ax.set_xticklabels(weather_types, rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Add a legend for weather colors
    handles = [
        plt.Line2D([0], [0], marker="o", color=color, label=weather, linestyle="")
        for weather, color in weather_colors.items()
    ]
    ax.legend(handles=handles, title="Weather Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Display the plot in Streamlit
    st.pyplot(fig)




    #########



    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from collections import Counter
    import streamlit as st

    # Streamlit app title
    st.title("Weather Descriptions Word Cloud")

    # Combine all weather descriptions into a single string
    weather_descriptions = pandas_df["Weather"].tolist()

    # Count occurrences of each weather description
    weather_counts = Counter(weather_descriptions)

    # Expand the descriptions proportional to their frequency
    weighted_weather_text = " ".join(
        [desc for desc, count in weather_counts.items() for _ in range(count)]
    )

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100  # Limit to the top 100 most frequent words for better diversity
    ).generate(weighted_weather_text)

    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")  # Hide axes
    ax.set_title("Diverse Word Cloud of Weather Descriptions", fontsize=16)

    # Display the plot in Streamlit
    st.pyplot(fig)



    ############



    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    # Streamlit app title
    st.title("KMeans Clustering of Cities Based on Weather")

    # Select relevant features for clustering
    features = pandas_df[["Temperature", "Humidity", "WindSpeed"]]

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust the number of clusters
    pandas_df["Cluster"] = kmeans.fit_predict(scaled_features)

    # Group the cities by clusters
    clustered_cities = pandas_df.groupby("Cluster")[["City", "Weather"]].agg(list).reset_index()

    # Display the clusters in Streamlit
    st.subheader("Clustered Cities")
    for index, row in clustered_cities.iterrows():
        st.write(f"**Cluster {row['Cluster']}**")
        st.write(f"Cities: {', '.join(row['City'])}")
        st.write(f"Weather Descriptions: {', '.join(row['Weather'])}")
        st.write("---")

    # Visualize clusters in 2D space
    st.subheader("Cluster Visualization")
    plt.figure(figsize=(10, 6))
    for cluster in range(4):
        cluster_data = pandas_df[pandas_df["Cluster"] == cluster]
        plt.scatter(
            cluster_data["Temperature"],
            cluster_data["Humidity"],
            label=f"Cluster {cluster}"
        )

    plt.title("City Clusters Based on Weather")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.grid()

    # Display the plot in Streamlit
    st.pyplot(plt)



    #####


    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import streamlit as st

    # Streamlit app title
    st.title("Grid of Word Clouds for Weather Descriptions by Clusters")

    # Create a dictionary for weather descriptions grouped by cluster
    cluster_weather = (
        pandas_df.groupby("Cluster")["Weather"]
        .apply(lambda descriptions: " ".join(descriptions))
        .to_dict()
    )

    # Determine the number of clusters
    num_clusters = len(cluster_weather)

    # Define the grid dimensions (4x4 as requested, or adjust dynamically)
    rows = (num_clusters // 2) + (num_clusters % 4 > 0)  # Rows based on total clusters
    cols = min(2, num_clusters)  # Max columns set to 4

    # Set up a grid layout using Streamlit columns
    st.subheader("Cluster Word Clouds in Grid Layout")
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))

    # Flatten axes array for consistent handling
    axes = axes.flatten()

    # Generate and display word clouds in the grid
    for idx, (cluster, weather_text) in enumerate(cluster_weather.items()):
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate(weather_text)

        # Display the word cloud in the corresponding grid cell
        axes[idx].imshow(wordcloud, interpolation="bilinear")
        axes[idx].axis("off")  # Hide axes
        axes[idx].set_title(f"Cluster {cluster}", fontsize=16)

    # Hide any unused subplots (if grid has more cells than clusters)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout and render the entire grid in Streamlit
    plt.tight_layout()
    st.pyplot(fig)



    #################



    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    # Streamlit app title
    st.title("Weather Data Visualizations by Region")
    st.subheader("Current Temperature by Region")

    # Sort the DataFrame by temperature for better visualization
    pandas_df_sorted = pandas_df.sort_values(by="Temperature", ascending=False)

    # Generate unique colors for each bar
    num_cities = len(pandas_df_sorted)
    colors = plt.cm.tab20(np.linspace(0, 1, num_cities))  # Use a colormap for diverse colors

    # Plot the bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(
        pandas_df_sorted["City"],
        pandas_df_sorted["Temperature"],
        color=colors,
        edgecolor="black"
    )

    # Add labels and title
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_xlabel("City", fontsize=12)
    ax.set_title("Current Temperature by Region", fontsize=16)
    ax.set_xticks(range(len(pandas_df_sorted["City"])))
    ax.set_xticklabels(pandas_df_sorted["City"], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with the temperature value
    for bar, temp in zip(bars, pandas_df_sorted["Temperature"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{temp:.1f}°C",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black"
        )

    # Display the chart in Streamlit
    st.pyplot(fig)

    st.subheader("Current Humidity by Region")

    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    # Streamlit app title

    # Sort the DataFrame for better visualization
    pandas_df_sorted = pandas_df.sort_values(by="Humidity", ascending=False)

    # Generate unique colors for each bar (for Humidity)
    num_cities = len(pandas_df_sorted)
    colors_humidity = plt.cm.tab20(np.linspace(0, 1, num_cities))  # Use a colormap for diverse colors

    # Plot the Humidity bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    bars_humidity = ax.bar(
        pandas_df_sorted["City"],
        pandas_df_sorted["Humidity"],
        color=colors_humidity,
        edgecolor="black"
    )

    # Add labels and title for Humidity chart
    ax.set_ylabel("Humidity (%)", fontsize=12)
    ax.set_xlabel("City", fontsize=12)
    ax.set_title("Humidity by Region", fontsize=16)
    ax.set_xticks(range(len(pandas_df_sorted["City"])))
    ax.set_xticklabels(pandas_df_sorted["City"], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with the humidity value
    for bar, humidity in zip(bars_humidity, pandas_df_sorted["Humidity"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{humidity:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black"
        )

    # Display the Humidity chart in Streamlit
    st.pyplot(fig)
    st.subheader("Current Wind Speed by Region")

    # Sort the DataFrame for Wind Speed visualization
    pandas_df_sorted_wind = pandas_df.sort_values(by="WindSpeed", ascending=False)

    # Generate unique colors for each bar (for Wind Speed)
    colors_wind = plt.cm.tab20(np.linspace(0, 1, num_cities))  # Use a colormap for diverse colors

    # Plot the Wind Speed bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    bars_wind = ax.bar(
        pandas_df_sorted_wind["City"],
        pandas_df_sorted_wind["WindSpeed"],
        color=colors_wind,
        edgecolor="black"
    )

    # Add labels and title for Wind Speed chart
    ax.set_ylabel("Wind Speed (m/s)", fontsize=12)
    ax.set_xlabel("City", fontsize=12)
    ax.set_title("Wind Speed by Region", fontsize=16)
    ax.set_xticks(range(len(pandas_df_sorted_wind["City"])))
    ax.set_xticklabels(pandas_df_sorted_wind["City"], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with the wind speed value
    for bar, wind_speed in zip(bars_wind, pandas_df_sorted_wind["WindSpeed"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{wind_speed:.1f} m/s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black"
        )

    # Display the Wind Speed chart in Streamlit
    st.pyplot(fig)

# Content for Tab 2
with tab2:
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




        
with tab3:
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