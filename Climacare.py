from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging

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
threshold_drought_temp = 25.5
threshold_drought_precip = 10.0
threshold_flood_precip = df['Precipitation'].mean() * 2

# Weather prediction function
def predict_weather(model_temp, model_precip, temperature, precipitation):
    pred_temp = model_temp.predict([[temperature, precipitation]])[0]
    pred_precip = model_precip.predict([[temperature, precipitation]])[0]
    return round(pred_temp, 2), round(pred_precip, 2)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location, temperature, precipitation = data['location'], float(data['temperature']), float(data['precipitation'])
    pred_temp, pred_precip = predict_weather(model_temp, model_precip, temperature, precipitation)

    if temperature > threshold_drought_temp and pred_precip < threshold_drought_precip:
        prediction = "Drought"
    elif temperature < threshold_drought_temp and pred_precip > threshold_flood_precip:
        prediction = "Flood"
    else:
        prediction = "Normal Weather Conditions"

    return jsonify({
        "location": location,
        "prediction": prediction,
        "predicted_temperature": pred_temp,
        "predicted_precipitation": pred_precip
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
    app.run(host='192.168.214.55', port=5000)
