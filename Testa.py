from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load and clean dataset
try:
    df = pd.read_csv("C:\\Users\\user\\Desktop\\2024\\Final Year Project\\Climacare\\NRBDataset\\Nairobi.csv")
    df = df[['Temperature', 'Precipitation']].dropna()
    df = df[(df['Temperature'] != 0) & (df['Precipitation'] != 0)]
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    exit()

# Train regression models
def train_models(df):
    X = df[['Temperature', 'Precipitation']]
    model_temp = RandomForestRegressor().fit(X, df['Temperature'])
    model_precip = RandomForestRegressor().fit(X, df['Precipitation'])
    return model_temp, model_precip

model_temp, model_precip = train_models(df)

# Define thresholds
threshold_drought_temp = 30
threshold_drought_precip = 0.5
threshold_flood_precip = 7.6

# Weather prediction function
def predict_weather(mean_temp, mean_precip):
    if mean_temp is not None and mean_temp > threshold_drought_temp and mean_precip <= threshold_drought_precip:
        prediction = "Drought"
    elif mean_temp is not None and mean_temp < threshold_drought_temp and mean_precip > threshold_flood_precip:
        prediction = "Flood"
    elif mean_precip > threshold_flood_precip:
        prediction = "Flood"
    else:
        prediction = "Normal Weather Conditions"

    return round(mean_temp, 2), round(mean_precip, 2), prediction

# Fetch weather data from the API using city name
def fetch_weather_data(api_key, city):
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecasts = data.get('list', [])
        if not forecasts:
            logger.error("No forecast data available.")
            return None
        
        # Extracting temperature and precipitation for each of the next 7 days
        daily_forecasts = []
        today = datetime.now().date()
        for i in range(7):
            forecast_date = today + timedelta(days=i)
            day_forecast = next((f for f in forecasts if datetime.fromtimestamp(f['dt']).date() == forecast_date), None)
            if day_forecast:
                temperature = day_forecast['main']['temp']
                precipitation = day_forecast.get('rain', {}).get('3h', 0)
                daily_forecasts.append({
                    "date": forecast_date.strftime('%Y-%m-%d'),
                    "predicted_temperature": temperature,
                    "predicted_precipitation": precipitation
                })
        
        return daily_forecasts
    else:
        logger.error(f"Error fetching weather data: {response.status_code}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    city = data['city']
    
    api_key = 'fc8713886530bd04c871c0e9a308d041'  # Your OpenWeatherMap API key
    daily_forecasts = fetch_weather_data(api_key, city)
    
    if daily_forecasts is None:
        return jsonify({"error": "Could not fetch weather data"}), 500

    mean_temp = sum([day['predicted_temperature'] for day in daily_forecasts]) / len(daily_forecasts)
    mean_precip = sum([day['predicted_precipitation'] for day in daily_forecasts]) / len(daily_forecasts)

    _, _, prediction = predict_weather(mean_temp, mean_precip)

    return jsonify({
        "city": city,
        "prediction": prediction,
        "daily_forecasts": daily_forecasts
    })

# Load chatbot patterns and responses into a dictionary
def load_chatbot_responses(file_path):
    chatbot_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    parts = line.strip().split('|')
                    if len(parts) == 2:
                        pattern, response = parts
                        chatbot_dict[pattern.strip().lower()] = response.strip()
                    else:
                        logger.warning(f"Skipping malformed line: {line}")
    except Exception as e:
        logger.error(f"Error loading chatbot responses: {e}")
    return chatbot_dict

chatbot_dict = load_chatbot_responses("C:\\Users\\user\\Desktop\\2024\\Final Year Project\\Main Draft\\Flask Server\\chatbot_responses.txt")
if not chatbot_dict:
    logger.error("No valid patterns found in chatbot responses file. Exiting.")
    exit()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message'].strip().lower()
    response = chatbot_dict.get(user_input, "I'm sorry, I didn't understand that. Please ask me a question related to floods, droughts, or disaster preparedness.")
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='192.168.150.54', port=5000)
