import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = None
model = None

def init_whisper():
    global processor, model
    if processor is None or model is None:
        # Load open-source Whisper model for multilingual transcription
        model_name = "openai/whisper-tiny" # Use 'base' or 'small' in production for better accuracy
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

def transcribe_and_analyze_audio(file_path: str):
    init_whisper()
    
    # 1. Extract Acoustic Biomarkers (Breathlessness indicators)
    # Load audio file with librosa
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Calculate some basic acoustic features that serve as biomarkers
    zcr = librosa.feature.zero_crossing_rate(audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    energy = librosa.feature.rms(y=audio)
    
    # Output as a numerical tensor (converted to float for JSON output)
    acoustic_features = {
        "mean_zero_crossing_rate": float(np.mean(zcr)),
        "mean_spectral_centroid": float(np.mean(spectral_centroid)),
        "mean_energy": float(np.mean(energy)),
        "tensor_shape": [1, 3] # Representing the extracted 1D tensor shape
    }
    
    # 2. Multilingual Transcription
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        # Whisper automatically detects the language and transcribes
        predicted_ids = model.generate(inputs.input_features)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription, acoustic_features
