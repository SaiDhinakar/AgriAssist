# AgriAssist - Plant Disease Detection API

AgriAssist is a FastAPI-based service designed to help farmers and gardeners identify plant diseases through image recognition and provide relevant treatment recommendations. It also integrates weather data analysis to assist in agricultural decision-making.

## Features

- Plant disease detection from images
- AI-powered disease explanations and treatment recommendations
- Real-time weather data for agricultural planning
- Warning and error tracking system

## API Endpoints

### 1. Plant Disease Detection

**Endpoint:** `POST /upload-image/`

Upload a plant image to detect diseases and get treatment recommendations.

#### Request

- Format: `multipart/form-data`
- Field: `file` (image file - JPG, PNG, etc.)

#### Response

```json
{
  "plant_disease": {
    "predicted_class": "Tomato_Late_blight",
    "class_index": 32,
    "probabilities": [0.001, 0.002, ..., 0.985, ...],
    "explanation": "Late blight is caused by the oomycete pathogen Phytophthora infestans that thrives in cool, wet conditions. It spreads rapidly through water splashes and wind. To manage this disease, apply copper-based fungicides early in the season, ensure proper plant spacing for air circulation, and water at the base of plants in the morning to allow foliage to dry quickly."
  }
}
```

### 2. Weather Information

**Endpoint:** `GET /fetch-current-weather-data`

Fetches current weather data for the configured location (default: Coimbatore).

#### Response

```json
{
  "location": {
    "name": "Coimbatore",
    "region": "Tamil Nadu",
    "country": "India"
  },
  "current": {
    "temp_c": 29.0,
    "temp_f": 84.2,
    "condition": {
      "text": "Partly cloudy",
      "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png"
    },
    "humidity": 65,
    "cloud": 25,
    "feelslike_c": 30.7,
    "feelslike_f": 87.3,
    "precip_mm": 0.0
  }
}
```

### 3. System Warnings

**Endpoint:** `GET /warnings/`

Retrieve system warnings and errors for monitoring.

#### Response

```json
{
  "logs": [
    {
      "timestamp": "2025-05-31T14:32:45.123456",
      "level": "WARNING",
      "message": "Invalid file type uploaded: text/plain"
    },
    {
      "timestamp": "2025-05-31T15:01:22.654321",
      "level": "ERROR",
      "message": "Processing error: Failed to load image"
    }
  ]
}
```

## Installation and Setup

### Prerequisites

- Python 3.12+
- TensorFlow 2.x
- FastAPI
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AgriAssist.git
cd AgriAssist
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:

```
WEATHER_API_KEY=your_weather_api_key
GEMINI_API_KEY=your_gemini_api_key
```

4. Run the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

When running the application, access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
