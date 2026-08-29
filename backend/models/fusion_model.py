import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

# ==========================================
# 1. Architecture: Sequence & Fusion Modules
# ==========================================

class SequenceProcessor(nn.Module):
    """
    RNN / LSTM Branch for processing sequential data like acoustic biomarkers
    over time. Outputs a dynamic fatigue hidden state.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SequenceProcessor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Return the final hidden state to represent accumulated fatigue
        return h_n[-1]

class FusionNetwork(nn.Module):
    """
    Fusion Gating Mechanism: Neural layer that dynamically weights and fuses 
    the latent outputs of the RNN branch and the tabular predictions.
    """
    def __init__(self, lstm_hidden_dim, tabular_pred_dim, output_dim=1):
        super(FusionNetwork, self).__init__()
        
        combined_dim = lstm_hidden_dim + tabular_pred_dim
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )
        
        # Fully connected regressor
        self.fc1 = nn.Linear(combined_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, lstm_out, tabular_out):
        # Concatenate latents
        combined = torch.cat((lstm_out, tabular_out), dim=1)
        
        # Apply gate (element-wise multiplication to weight feature importance dynamically)
        gated = combined * self.gate(combined)
        
        # Output prediction
        x = self.relu(self.fc1(gated))
        x = self.fc2(x)
        return x


# ==========================================
# 2. Physics-Informed Loss Function
# ==========================================

class PhysicsInformedLoss(nn.Module):
    """
    Custom PyTorch loss function combining standard MSE with a biomechanical penalty.
    Penalizes predictions where calculated energy expenditure violates physiological limits.
    """
    def __init__(self, max_kcal_per_kg_min=0.35):
        # 0.35 kcal/kg/min is approximately 21 METs, near the absolute peak of human performance.
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.max_kcal_per_kg_min = max_kcal_per_kg_min
        
    def forward(self, predictions, targets, weights, duration_mins):
        """
        predictions: Predicted calories burned (batch_size, 1)
        targets: Ground truth calories burned (batch_size, 1)
        weights: Body weight in kg (batch_size, 1)
        duration_mins: Activity duration in minutes (batch_size, 1)
        """
        base_loss = self.mse(predictions, targets)
        
        # Biomechanical penalty
        max_possible_cals = weights * duration_mins * self.max_kcal_per_kg_min
        
        # ReLU allows only positive differences (i.e., physiological limit violations) to be penalized
        violation = torch.relu(predictions - max_possible_cals)
        
        # Squared penalty for violating the boundary
        physics_penalty = torch.mean(violation ** 2)
        
        # Weighted combination (lambda can be adjusted)
        lambda_penalty = 10.0 
        return base_loss + (lambda_penalty * physics_penalty)


# ==========================================
# 3. Model Pipeline & Interface
# ==========================================

class NeuroMetabolicFusionNetwork:
    """
    The main wrapper handling both the XGBoost branch (Tabular) and the PyTorch branch (Sequence/Fusion).
    """
    def __init__(self, seq_input_dim=3, lstm_hidden_dim=32):
        self.seq_processor = SequenceProcessor(seq_input_dim, lstm_hidden_dim)
        self.fusion_net = FusionNetwork(lstm_hidden_dim=lstm_hidden_dim, tabular_pred_dim=1)
        
        # Gradient Boosting Branch for static biometric/contextual features
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        
        # Setting PyTorch models to eval mode for inference
        self.seq_processor.eval()
        self.fusion_net.eval()
        
    def _preprocess_tabular(self, features_dict):
        # Default mock values if features are missing
        weight = features_dict.get('weight_kg', 70.0)
        met_value = features_dict.get('met_value', 1.5) # Base Non-Exercise MET
        temp = features_dict.get('temperature', 20.0)
        humidity = features_dict.get('humidity', 50.0)
        
        return np.array([[weight, met_value, temp, humidity]])

    def _preprocess_sequence(self, features_dict):
        # Extract sequence features (e.g. acoustic biomarkers over time)
        # Expected shape: List of lists, e.g., [[ZCR, Centroid, Energy], ...]
        biomarkers = features_dict.get('acoustic_biomarkers', [[0.0, 0.0, 0.0]])
        
        # If it's a flat dict of means from our audio_service, convert to sequence
        if isinstance(biomarkers, dict):
            biomarkers = [[
                biomarkers.get('mean_zero_crossing_rate', 0.0),
                biomarkers.get('mean_spectral_centroid', 0.0),
                biomarkers.get('mean_energy', 0.0)
            ]]
            
        # Shape: (batch_size=1, seq_length, input_dim)
        return torch.tensor([biomarkers], dtype=torch.float32)
        
    def predict_neat(self, features_dict):
        """
        Clean inference method ready for FastAPI integration.
        """
        # 1. Gradient Boosting Branch (Tabular Data)
        tab_features = self._preprocess_tabular(features_dict)
        
        try:
            # Predict using XGBoost (will throw NotFittedError if untrained)
            tabular_pred = self.xgb_model.predict(tab_features)
        except Exception:
            # Fallback mock prediction to prevent API crash on inference before training
            duration_hr = features_dict.get('duration_mins', 10.0) / 60.0
            weight = features_dict.get('weight_kg', 70.0)
            met = features_dict.get('met_value', 1.5)
            mock_burn = weight * met * duration_hr
            tabular_pred = np.array([mock_burn])
            
        tabular_pred_tensor = torch.tensor(tabular_pred, dtype=torch.float32).unsqueeze(1) # Shape: (1, 1)

        # 2. RNN / LSTM Branch (Sequence Data)
        seq_features = self._preprocess_sequence(features_dict)
        
        with torch.no_grad():
            lstm_latent = self.seq_processor(seq_features)
            
            # 3. Fusion Gating Mechanism
            final_calories = self.fusion_net(lstm_latent, tabular_pred_tensor)
            
        raw_calories = float(final_calories.item())
        
        # Physics Constraint / Lower Bound
        weight_kg = features_dict.get('weight_kg', 70.0)
        met_value = features_dict.get('met_value', 1.5)
        duration_mins = features_dict.get('duration_mins', 30.0)
        
        # Base Calories = (MET * 3.5 * weight_kg / 200) * duration_mins
        base_calories = (met_value * 3.5 * weight_kg / 200) * duration_mins
        
        # Clamp the output to ensure the neural net doesn't hallucinatively predict negative or physically impossible lows
        final_predicted_calories = max(raw_calories, base_calories)
            
        return {
            "predicted_calories": float(final_predicted_calories),
            "baseline_tabular_prediction": float(tabular_pred[0])
        }

# Instantiate a global pipeline for direct import in main.py
nmfn_pipeline = NeuroMetabolicFusionNetwork()

def predict_neat(features_dict):
    """
    Exposed functional interface for FastAPI.
    Example payload:
    {
        "weight_kg": 75,
        "met_value": 2.5,
        "temperature": 28,
        "humidity": 65,
        "duration_mins": 30,
        "acoustic_biomarkers": [[0.1, 1500, 0.05], [0.12, 1600, 0.06]]
    }
    """
    return nmfn_pipeline.predict_neat(features_dict)
