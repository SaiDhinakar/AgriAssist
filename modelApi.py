from fastapi import FastAPI
import os 
from dotenv import dotenv_values
import joblib
from sklearn.preprocessing import OneHotEncoder
from WeatherAnalysis import WeatherAnalyzer
import google.generativeai as genai
import numpy as np
import pandas as pd

app = FastAPI()

WEATHER_API = dotenv_values(".env").get("WEATHER_API_KEY")
CHAT_BOT_API = dotenv_values(".env").get("GEMINI_API_KEY")

MODEL_PATH = os.path.abspath("models")

PEST_RISK_SUGGESSTION_MODEL = None
DISEASE_PREDICTION_MODEL = None

city = "Coimbatore"
    
analyzer = WeatherAnalyzer(WEATHER_API, city)

pest_data_processed = pd.read_csv(os.path.join(os.path.abspath("Datasets"), "pest_.csv"))

plant_feature_names = None   

def preprocessing():
    global plant_feature_names

    def extract_humidity_avg(humidity_range):
        low, high = map(int, humidity_range.split('-'))
        return (low + high) / 2
    
    # Create a mapping dictionary for rain values to percentages
    rain_mapping = {
        'Low': 20,       # Low rain -> 20% chance
        'Moderate': 50,  # Moderate rain -> 50% chance
        'High': 80       # High rain -> 80% chance
    }

    #  a new column with numerical percentage values
    pest_data_processed['rain_percentage'] = pest_data_processed['rain'].map(rain_mapping)
    
    pest_data_processed['humidity_avg'] = pest_data_processed['humidity'].apply(extract_humidity_avg)

    # Create features (X) and target (y)
    X = pest_data_processed[['plant', 'avg_temp', 'humidity_avg', 'rain_percentage']]
    y = pest_data_processed['insect_name']

    # One-hot encode the plant names
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    plant_encoded = encoder.fit_transform(X[['plant']])
    plant_feature_names = encoder.get_feature_names_out(['plant'])

preprocessing()

def load_models():
    # Load the trained pest prediction model
    global PEST_RISK_SUGGESSTION_MODEL, DISEASE_PREDICTION_MODEL
    try:
        # Load the model from file
        print(os.path.join(MODEL_PATH,'pest_prediction_model.joblib'))
        model = joblib.load(os.path.join(MODEL_PATH,'pest_prediction_model.joblib'))
        print("Pest prediction model loaded successfully")
        
        # Display model information
        print(f"Model type: {type(model).__name__}")
        # print(f"Number of trees in forest: {model.n_estimators}")
        print(f"Number of insect classes: {len(model.classes_)}")
        print(f"Insect classes: {', '.join(model.classes_[:5])}...")
        
        PEST_RISK_SUGGESSTION_MODEL = model
        
    except FileNotFoundError:
        print("Error: Model file not found.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

load_models()

def setup_chat_bot():
    """Configure and return the Gemini model"""
    genai.configure(api_key=CHAT_BOT_API)
    # For free API key, use the standard Gemini model
    # Use the Gemini 2.0 Flash model
    model_name = 'models/gemini-2.0-flash'
    print(f"Using model: {model_name}")
    return genai.GenerativeModel(model_name)

def process_user_query(query):
    
    model = setup_chat_bot()
    
    prompt = f"""

    USER QUERY:
    {query}
    
    Provide a detailed and helpful response in a proper format that could be simple and understandable.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# Function to predict insects for a given plant based on current weather
def predict_pest_risk(plant_name, current_weather):
    model = PEST_RISK_SUGGESSTION_MODEL
    print(current_weather)
    # Extract relevant weather info
    temp = current_weather.get('temp', 0)
    humidity = current_weather.get('humidity', 0)
    rain_chance = current_weather.get('rain_chance', 0)
    
    # Prepare input for prediction
    plant_input = np.zeros((1, len(plant_feature_names)))
    try:
        plant_idx = np.where(plant_feature_names == f'plant_{plant_name}')[0][0]
        plant_input[0, plant_idx] = 1
    except:
        print(f"Warning: Plant '{plant_name}' not found in training data")
        
    weather_input = np.array([[temp, humidity, rain_chance]])
    input_data = np.hstack([plant_input, weather_input])
    
    # Get probabilities for each insect
    probas = model.predict_proba(input_data)[0]
    
    # Get top 3 insects with highest probabilities
    top_indices = probas.argsort()[-3:][::-1]
    top_insects = [(model.classes_[i], probas[i]) for i in top_indices]
    
    # Get damage descriptions for these insects
    results = []
    for insect, prob in top_insects:
        damage = pest_data_processed[pest_data_processed['insect_name'] == insect]['damage'].values[0]
        results.append({
            'insect': insect,
            'probability': prob * 100,  # Convert to percentage
            'damage': damage
        })
    print(results)
    return results


@app.get("/warnings")
def warnings():
    warnings = {
        "Warning":"No warnings!"
    }
    if not PEST_RISK_SUGGESSTION_MODEL:
        warnings["Warning"]="Pest risk suggestion model not loaded!"
    elif not DISEASE_PREDICTION_MODEL:
        warnings["Warning"]="Disease prediction model not loaded!"
    
    return warnings

@app.get("/fetch-current-weather-data")
def fetch_current_weather_data():
    return analyzer.fetch_current_weather()

@app.get("/chat/")
def chat(query: str):
    return {
        "Response": process_user_query(query)
    }

@app.get("/predict/{query}")
def predict_pest(query: str):
    response = predict_pest_risk(query, fetch_current_weather_data())
    return response

