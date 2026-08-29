from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import json
from typing import List, Optional
import requests

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from services.audio_service import transcribe_and_analyze_audio
from services.nlp_service import extract_entities
from services.weather_service import get_weather_data
import torch
import numpy as np
from ml_engine import NeuroMetabolicFusionNetwork, get_shap_values

app = FastAPI(title="Context Fusion Hub - NEAT Tracker")

# Initialize the PyTorch Fusion model
fusion_model = NeuroMetabolicFusionNetwork(tabular_dim=4, seq_input_dim=3, hidden_dim=16)
try:
    fusion_model.load_state_dict(torch.load('fusion_model.pth', weights_only=True))
except Exception as e:
    print(f"Warning: Failed to load model weights: {e}")
fusion_model.eval()

# Dummy background tabular for SHAP initialization
np.random.seed(42)
fusion_X_train = torch.tensor(np.random.uniform(10, 100, (1000, 4)), dtype=torch.float32)

# Initialize Gemini Client (uses GEMINI_API_KEY environment variable)
try:
    client = genai.Client()
except Exception as e:
    client = None
    print(f"Warning: Failed to initialize Gemini Client: {e}")

class TextPayload(BaseModel):
    text: str

class NeatContextPayload(BaseModel):
    weight_kg: float = 70.0
    met_value: float = 1.5
    temperature: float = 20.0
    humidity: float = 50.0
    duration_mins: float = 30.0
    acoustic_biomarkers: List[List[float]] = [[0.0, 0.0, 0.0]]
    current_fatigue_level: float = 30.0
    intensity: str = 'Normal'
    transcript: str = ''
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class LLMMetricsOutput(BaseModel):
    duration_mins: int
    base_met: float
    detected_activity: str

async def extract_metrics_with_llm(transcript: str, default_duration: float) -> LLMMetricsOutput:
    """
    Uses a Large Language Model to semantically parse the Tanglish/English transcript.
    """
    if not client or not transcript.strip():
        return LLMMetricsOutput(
            duration_mins=int(default_duration),
            base_met=1.3,
            detected_activity="resting (fallback)"
        )

    prompt = f"""You are an expert Biomechanical Data Scientist and a native speaker of English, Tamil, and colloquial Tanglish. Your job is to extract highly accurate Metabolic Equivalent of Task (MET) scores and durations from unstructured, messy user inputs.

Core Directives:

Deep Semantic Extraction: Read the input naturally. Understand all tenses, slang, typos, and mixed languages to deduce the precise physical reality of what the user is doing.

Dynamic MET Retrieval: Do not use any hardcoded lists. Query your deep internal knowledge of the Compendium of Physical Activities. Dynamically determine and assign the exact, scientifically accurate MET score for the specific deduced activity.

Contextual Stacking: If the user describes overlapping activities (e.g., walking while talking, sitting but vigorously shaking legs, or pacing while carrying a heavy object), dynamically calculate a combined, highly accurate MET score based on biomechanical realities.

If duration is not mentioned, default to {int(default_duration)} minutes.

Output:
Output ONLY a raw, structured JSON object: {{"duration_mins": int, "base_met": float, "detected_activity": "clear english summary of the deduced activity"}}.

Transcript: "{transcript}"
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMMetricsOutput,
                temperature=0.1
            ),
        )
        
        data = json.loads(response.text)
        return LLMMetricsOutput(**data)
        
    except Exception as e:
        print(f"LLM Semantic Parsing failed: {e}")
        return LLMMetricsOutput(
            duration_mins=int(default_duration),
            base_met=1.3,
            detected_activity="resting (error fallback)"
        )


def get_live_weather(lat: Optional[float] = None, lon: Optional[float] = None):
    # Default thermal baseline for Arcot
    default_temp = 30.0
    default_humidity = 60.0
    
    if lat is None or lon is None:
        return default_temp, default_humidity
        
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return default_temp, default_humidity
        
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        temp = data.get('main', {}).get('temp', default_temp)
        humidity = data.get('main', {}).get('humidity', default_humidity)
        return float(temp), float(humidity)
    except Exception as e:
        print(f"Weather API failed: {e}")
        return default_temp, default_humidity

import math

def calculate_fatigue_curve(calories_burned: float):
    curve = []
    base = 20.0
    # The more you burn, the steeper the fatigue climbs over 4 hours
    steepness = 0.5 + (calories_burned / 400.0) 
    for hour in range(5):
        # Exponential climb towards 100% fatigue
        fatigue = base + (80.0 * (1.0 - math.exp(-steepness * (hour / 2.0))))
        curve.append(min(100, int(round(fatigue))))
    return curve

@app.post("/calculate_neat")
async def calculate_neat(payload: NeatContextPayload):
    """
    Stage 4: Predictive Outputs & Visualization.
    The ultimate endpoint that accepts gathered multimodal context,
    runs the NMFN predict_neat function, and generates visualization structures.
    """
    features_dict = payload.dict()
    
    transcript = features_dict.get('transcript', '')
    intensity = features_dict.get('intensity', 'Normal')
    fallback_duration = features_dict.get('duration_mins', 10.0)
    
    # 1. Semantic Extraction via LLM
    llm_metrics = await extract_metrics_with_llm(transcript, fallback_duration)
    
    # Apply Intensity Multiplier to Base MET
    intensity_mult = 1.0
    if intensity.lower() == 'low':
        intensity_mult = 0.8
    elif intensity.lower() == 'high':
        intensity_mult = 1.2
        
    dynamic_met = llm_metrics.base_met * intensity_mult
    
    # Overwrite payload values with intelligent LLM truth
    features_dict['met_value'] = dynamic_met
    features_dict['duration_mins'] = float(llm_metrics.duration_mins)
    
    # 2. Integrate Live GPS Meteorological Data
    lat = features_dict.get('latitude')
    lon = features_dict.get('longitude')
    live_temp, live_humidity = get_live_weather(lat, lon)
    features_dict['temperature'] = live_temp
    features_dict['humidity'] = live_humidity
    
    # 3. Run the PyTorch Neuro-Metabolic Fusion Network
    tabular_tensor = torch.tensor([[features_dict['weight_kg'], features_dict['met_value'], features_dict['temperature'], features_dict['humidity']]], dtype=torch.float32)
    seq_tensor = torch.rand((1, 5, 3), dtype=torch.float32) # Dummy historical RNN sequence
    
    with torch.no_grad():
        prediction_tensor = fusion_model(tabular_tensor, seq_tensor)
        calories_rate = prediction_tensor.item()
        
    # Scale calories by the actual activity duration (model output is per hour)
    calories_burned = calories_rate * (features_dict['duration_mins'] / 60.0)
    
    # 4. Explainable AI (SHAP)
    dynamic_shap_array = get_shap_values(
        weight=features_dict['weight_kg'],
        met=features_dict['met_value'],
        temp=features_dict['temperature'],
        humidity=features_dict['humidity'],
        model=fusion_model,
        X_train=fusion_X_train
    )
    
    feature_names = ["Weight", "MET", "Temp", "Humid"] # Shortened for dashboard
    
    shap_graph_data = {
        "labels": feature_names,
        "datasets": [
            {
                "data": dynamic_shap_array
            }
        ]
    }
    
    # 5. Circadian Fatigue Forecaster
    fatigue_curve = calculate_fatigue_curve(calories_burned)
    
    return {
        "status": "success",
        "data": {
            "calories_burned": round(calories_burned, 1),
            "dynamic_met_used": round(dynamic_met, 2),
            "detected_activity": llm_metrics.detected_activity,
            "parsed_duration_mins": llm_metrics.duration_mins,
            "temperature": live_temp,
            "humidity": live_humidity,
            "shap_graph_data": shap_graph_data,
            "fatigue_curve": fatigue_curve
        }
    }

from fastapi import Form

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    weight: float = Form(70.0),
    intensity: str = Form("Normal"),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None)
):
    """
    Multilingual Acoustic Stream (Multimodal LLM Integration)
    Accepts audio, uploads to Gemini, extracts MET metrics, and runs the full predictive pipeline.
    """
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if not client:
            return {"status": "error", "message": "Gemini client not initialized"}
            
        # Upload the audio file to Gemini servers
        uploaded_file = client.files.upload(file=temp_file_path)
        
        prompt = f"""You are an expert Biomechanical Data Scientist and a native speaker of English, Tamil, and colloquial Tanglish. Your job is to extract highly accurate Metabolic Equivalent of Task (MET) scores and durations from unstructured, messy user inputs.

Core Directives:

Deep Semantic Extraction: Listen to the audio naturally. Understand all tenses, slang, typos, and mixed languages to deduce the precise physical reality of what the user is doing.

Dynamic MET Retrieval: Do not use any hardcoded lists. Query your deep internal knowledge of the Compendium of Physical Activities. Dynamically determine and assign the exact, scientifically accurate MET score for the specific deduced activity.

Contextual Stacking: If the user describes overlapping activities (e.g., walking while talking, sitting but vigorously shaking legs, or pacing while carrying a heavy object), dynamically calculate a combined, highly accurate MET score based on biomechanical realities.

If duration is not mentioned, default to 10 minutes.

Output:
Output ONLY a raw, structured JSON object: {{"duration_mins": int, "base_met": float, "detected_activity": "clear english summary of the deduced activity"}}.
"""
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMMetricsOutput,
                temperature=0.1
            ),
        )
        
        llm_metrics = json.loads(response.text)
        
        # Cleanup uploaded file from Google servers
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception as e:
            print(f"Warning: Failed to delete remote file {uploaded_file.name}: {e}")
            
        # Apply Intensity Multiplier to Base MET
        intensity_mult = 1.0
        if intensity.lower() == 'low':
            intensity_mult = 0.8
        elif intensity.lower() == 'high':
            intensity_mult = 1.2
            
        dynamic_met = llm_metrics["base_met"] * intensity_mult
        
        # Integrate Live GPS Meteorological Data
        live_temp, live_humidity = get_live_weather(latitude, longitude)
        
        # Build features dict for NMFN
        features_dict = {
            "weight_kg": weight,
            "met_value": dynamic_met,
            "temperature": live_temp,
            "humidity": live_humidity,
            "duration_mins": float(llm_metrics["duration_mins"]),
            "acoustic_biomarkers": [[0.1, 1500, 0.05]],
            "current_fatigue_level": 40.0
        }
        
        # Run PyTorch NMFN
        tabular_tensor = torch.tensor([[features_dict['weight_kg'], features_dict['met_value'], features_dict['temperature'], features_dict['humidity']]], dtype=torch.float32)
        seq_tensor = torch.rand((1, 5, 3), dtype=torch.float32) # Dummy historical RNN sequence
        
        with torch.no_grad():
            prediction_tensor = fusion_model(tabular_tensor, seq_tensor)
            calories_rate = prediction_tensor.item()
            
        # Scale calories by the actual activity duration (model output is per hour)
        calories_burned = calories_rate * (features_dict['duration_mins'] / 60.0)
        
        dynamic_shap_array = get_shap_values(
            weight=features_dict['weight_kg'],
            met=features_dict['met_value'],
            temp=features_dict['temperature'],
            humidity=features_dict['humidity'],
            model=fusion_model,
            X_train=fusion_X_train
        )
        
        feature_names = ["Weight", "MET", "Temp", "Humid"] # Shortened for dashboard
        
        shap_graph_data = {
            "labels": feature_names,
            "datasets": [
                {
                    "data": dynamic_shap_array
                }
            ]
        }
        
        fatigue_curve = calculate_fatigue_curve(calories_burned)
            
        return {
            "status": "success",
            "data": {
                "duration_mins": llm_metrics["duration_mins"],
                "base_met": llm_metrics["base_met"],
                "detected_activity": llm_metrics["detected_activity"],
                "calories_burned": round(calories_burned, 1),
                "dynamic_met_used": round(dynamic_met, 2),
                "temperature": live_temp,
                "humidity": live_humidity,
                "shap_graph_data": shap_graph_data,
                "fatigue_curve": fatigue_curve
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/process_text")
async def process_text(payload: TextPayload):
    """
    Multilingual NLP Stream
    """
    text = payload.text
    entities = extract_entities(text)
    
    weather_data = None
    environment = entities.get("Environment")
    if environment:
        weather_data = await get_weather_data(environment)
        
    return {
        "status": "success",
        "data": {
            "text": text,
            "extracted_entities": entities,
            "environmental_context": weather_data
        }
    }
