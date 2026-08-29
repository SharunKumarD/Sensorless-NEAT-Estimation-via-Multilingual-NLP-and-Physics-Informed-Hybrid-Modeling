from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import json
from typing import List

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from services.audio_service import transcribe_and_analyze_audio
from services.nlp_service import extract_entities
from services.weather_service import get_weather_data
from services.explainability_service import generate_shap_explainability, forecast_circadian_fatigue
from models.fusion_model import nmfn_pipeline, predict_neat

app = FastAPI(title="Context Fusion Hub - NEAT Tracker")

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
    
    # 2. Run the Neuro-Metabolic Fusion Network
    prediction_result = predict_neat(features_dict)
    calories_burned = prediction_result.get("predicted_calories", 0.0)
    
    # 3. Explainable AI (SHAP)
    tabular_features = nmfn_pipeline._preprocess_tabular(features_dict)
    feature_names = ["Weight", "MET (Pacing)", "Temperature", "Humidity"]
    
    shap_graph_data = generate_shap_explainability(
        xgb_model=nmfn_pipeline.xgb_model,
        features_array=tabular_features,
        feature_names=feature_names
    )
    
    # 4. Circadian Fatigue Forecaster
    circadian_forecast_curve = forecast_circadian_fatigue(
        base_fatigue_level=payload.current_fatigue_level,
        hours=4,
        resolution_mins=60
    )
    
    return {
        "status": "success",
        "data": {
            "calories_burned": round(calories_burned, 1),
            "dynamic_met_used": round(dynamic_met, 2),
            "detected_activity": llm_metrics.detected_activity,
            "parsed_duration_mins": llm_metrics.duration_mins,
            "shap_graph_data": shap_graph_data,
            "circadian_forecast_curve": circadian_forecast_curve
        }
    }

from fastapi import Form

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    weight: float = Form(70.0),
    intensity: str = Form("Normal")
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
        
        # Build features dict for NMFN
        features_dict = {
            "weight_kg": weight,
            "met_value": dynamic_met,
            "temperature": 24.0,
            "humidity": 60.0,
            "duration_mins": float(llm_metrics["duration_mins"]),
            "acoustic_biomarkers": [[0.1, 1500, 0.05]],
            "current_fatigue_level": 40.0
        }
        
        # Run NMFN
        prediction_result = predict_neat(features_dict)
        calories_burned = prediction_result.get("predicted_calories", 0.0)
        
        tabular_features = nmfn_pipeline._preprocess_tabular(features_dict)
        feature_names = ["Weight", "MET (Pacing)", "Temperature", "Humidity"]
        
        shap_graph_data = generate_shap_explainability(
            xgb_model=nmfn_pipeline.xgb_model,
            features_array=tabular_features,
            feature_names=feature_names
        )
        
        circadian_forecast_curve = forecast_circadian_fatigue(
            base_fatigue_level=features_dict["current_fatigue_level"],
            hours=4,
            resolution_mins=60
        )
            
        return {
            "status": "success",
            "data": {
                "duration_mins": llm_metrics["duration_mins"],
                "base_met": llm_metrics["base_met"],
                "detected_activity": llm_metrics["detected_activity"],
                "calories_burned": round(calories_burned, 1),
                "dynamic_met_used": round(dynamic_met, 2),
                "shap_graph_data": shap_graph_data,
                "circadian_forecast_curve": circadian_forecast_curve
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
