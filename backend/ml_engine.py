import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression

# ==========================================
# STAGE 3: Neuro-Metabolic Fusion Network
# ==========================================

class NeuroMetabolicFusionNetwork(nn.Module):
    """
    A multi-branch PyTorch architecture that fuses static biomechanical data
    with sequential time-series sensor data.
    """
    def __init__(self, tabular_dim=4, seq_input_dim=3, hidden_dim=16):
        super(NeuroMetabolicFusionNetwork, self).__init__()
        
        # Branch 1: Dense Linear Layer for Tabular (Weight, MET, Temp, Humidity)
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Branch 2: RNN Layer for Historical Sequence Data
        self.rnn = nn.RNN(input_size=seq_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.seq_linear = nn.Linear(hidden_dim, 8)
        
        # Fusion Gate: Concatenates both branches
        self.fusion_gate = nn.Sequential(
            nn.Linear(8 + 8, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Final predictive output (Calories Burned)
        )
        
    def forward(self, tabular_x, seq_x):
        # 1. Process tabular features
        tab_out = self.tabular_branch(tabular_x)
        
        # 2. Process sequential features
        _, h_n = self.rnn(seq_x)
        seq_out = self.seq_linear(h_n.squeeze(0)) # Get last hidden state
        
        # 3. Concatenate at Fusion Gate
        fused = torch.cat((tab_out, seq_out), dim=1)
        
        # 4. Final Output
        return self.fusion_gate(fused)

def physics_penalty_function(predictions, max_limit=1500.0):
    """
    Physics-Informed Neural Network (PINN) Constraint:
    Heavily penalizes predictions that exceed human biomechanical caloric limits
    (e.g., > 1500 kcal burned per hour).
    """
    # ReLU ensures we only apply a penalty if pred > max_limit
    violation = torch.relu(predictions - max_limit)
    # Return Mean Squared Error of the violations
    return torch.mean(violation ** 2)

def initialize_model():
    """
    Generates dummy sequence & tabular data, trains the Fusion Network
    with a physics-informed custom loss, and saves the compiled PyTorch model.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    seq_length = 5
    seq_input_dim = 3 # e.g., [heart_rate, acoustic_pitch, acceleration]
    tabular_dim = 4
    
    print("Generating Tabular & Sequence Dummy Datasets...")
    # Tabular Data: [Weight, MET, Temperature, Humidity]
    weights = np.random.uniform(50, 100, n_samples)
    mets = np.random.uniform(1, 10, n_samples)
    temps = np.random.uniform(10, 40, n_samples)
    humidities = np.random.uniform(30, 90, n_samples)
    
    # Ground Truth: Rough caloric estimation
    y = (weights * mets * 0.0175 * 60) + (temps * 0.5) + (humidities * 0.1)
    
    tabular_data = np.column_stack([weights, mets, temps, humidities])
    
    # Sequence Data
    seq_data = np.random.rand(n_samples, seq_length, seq_input_dim)
    
    # Convert to PyTorch Tensors
    X_tab_t = torch.tensor(tabular_data, dtype=torch.float32)
    X_seq_t = torch.tensor(seq_data, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Dataloader
    dataset = TensorDataset(X_tab_t, X_seq_t, y_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Compile Model
    model = NeuroMetabolicFusionNetwork(tabular_dim, seq_input_dim, hidden_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss_fn = nn.MSELoss()
    
    epochs = 20
    lambda_penalty = 10.0 # High multiplier for physics violations
    max_caloric_limit = 1500.0 
    
    print("Training PyTorch Neuro-Metabolic Fusion Network...")
    model.train()
    for epoch in range(epochs):
        for batch_tab, batch_seq, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(batch_tab, batch_seq)
            
            # Physics-Informed Custom Loss
            mse_loss = mse_loss_fn(predictions, batch_y)
            physics_loss = physics_penalty_function(predictions, max_limit=max_caloric_limit)
            
            loss = mse_loss + (lambda_penalty * physics_loss)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
    # Save Model
    model_path = 'fusion_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to {model_path}")
    
    model.eval()
    return model, X_tab_t

def get_shap_values(weight, met, temp, humidity, model, X_train):
    """
    Extracts dynamic feature importance from the PyTorch model using shap.GradientExplainer.
    Ensures the output array cleanly sums to 100 for the React Native dashboard.
    """
    try:
        # Create a dummy background sequence tensor matching X_train's batch size
        background_seq = torch.zeros(X_train.shape[0], 5, 3)
        
        # Initialize GradientExplainer with multi-input list
        explainer = shap.GradientExplainer(model, [X_train, background_seq])
        
        # Prepare live input tensors
        live_tab = torch.tensor([[weight, met, temp, humidity]], dtype=torch.float32)
        live_seq = torch.zeros(1, 5, 3)
        
        # Calculate SHAP values for the live instance
        # shap_values is a list of arrays, one per input. We want tabular branch (index 0).
        shap_vals = explainer.shap_values([live_tab, live_seq])[0][0]
    except Exception as e:
        print(f"SHAP Error: {e}")
        shap_vals = np.array([40, 30, 15, 15]) # Fallback
        
    abs_shap = np.abs(shap_vals)
    total = np.sum(abs_shap)
    
    if total == 0:
        return [25, 25, 25, 25]
        
    percentages = (abs_shap / total) * 100
    int_percentages = np.round(percentages).astype(int)
    
    # Adjust for fractional rounding errors
    diff = 100 - np.sum(int_percentages)
    if diff != 0:
        max_idx = np.argmax(int_percentages)
        int_percentages[max_idx] += diff
        
    return int_percentages.tolist()

if __name__ == '__main__':
    initialize_model()
