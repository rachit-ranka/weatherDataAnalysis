# Weather Data Analysis

## Overview
The **Weather Data Analysis** project aims to provide real-time visualizations for various weather parameters such as wind speed, temperature, humidity, snowfall, and rainfall. The project fetches weather data using the OpenWeather API and generates insightful visualizations to help users understand weather trends across different regions. It also uses historical data to provide useful visualizations about various parameters. The user can also predict the temperature of a particular place. 

## Features
- **Real-time Weather Data Retrieval:** Fetches live weather data from the OpenWeather API.
- **Interactive Visualizations:** Displays wind speed, temperature, humidity, snowfall, and rainfall.
- **Regional Weather Mapping:** Provides a geographic view of weather conditions across multiple regions.
- **Severity Analysis:** Highlights critical weather conditions based on real-time data.
- **Word Cloud Generation:** Analyzes textual weather data to generate word clouds.
- **Historical Weather Analysis:** Analyzes historical weather data to various visualizatons.
- **Weather Forecasting for various places:** Based on the trained LSTM model it gives temperature prediction for different locations.

<video width="600" controls>
  <source src="https://raw.githubusercontent.com/rachit-ranka/weatherDataAnalysis/preview/realtime analysis.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
<video width="600" controls>
  <source src="https://raw.githubusercontent.com/rachit-ranka/weatherDataAnalysis/preview/historical and prediction.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.11
- Required Python libraries (listed in `requirements.txt`)

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/rachit-ranka/weatherDataAnalysis.git
   cd weatherDataAnalysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up OpenWeather API key:
   - Sign up on [OpenWeather](https://openweathermap.org/) to get an API key.
   - Add api_key to streamlit/app-merged.py
     ```app-merged.py
     API_KEY = "Your_API_Key"
     ```
4. Set up MongoDB
   - Import the dataset to MongoDB
   - Set up mongoDB client in app-merged.py
     ```app-merged.py
     client = pymongo.MongoClient("Enter_your_LocalHost_id")
     ```
   
6. Run the script:
   ```bash
   streamlit run app-merged.py
   ```

## Usage
- The script retrieves weather data for specified regions and generates visualizations.
- Adjust parameters in the script to analyze different locations and weather conditions.
- Output includes charts, maps, and summary statistics.

## File Structure
```
weatherDataAnalysis/
│── dataset/            # Weather data files
│── Forecasting_model/  # Python scripts for forecasting model
│── streamlit/          # Python scripts for streamlit app
│── requirements.txt    # Dependencies
│── BDA Final.ipynb     # Visualizations
│── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Contact
For any queries, reach out to **Rachit Ranka** via [GitHub](https://github.com/rachit-ranka).
