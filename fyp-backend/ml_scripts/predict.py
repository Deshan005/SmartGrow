import torch
import torch.nn as nn
import sys
import numpy as np

class DeeperWaterLevelNN(nn.Module):
    def __init__(self, input_features=13, dropout_rate=0.0):
        super(DeeperWaterLevelNN, self).__init__()
        self.layer1 = nn.Linear(input_features, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Load the trained model
model = DeeperWaterLevelNN(input_features=13)
model.load_state_dict(torch.load('ml_scripts/water_level_model.pth'))
model.eval()

# Get input arguments
plant_type = sys.argv[1]  # "Chilli" or "Brinjal"
height_cm = float(sys.argv[2])
temperature_c = float(sys.argv[3])
humidity_percent = float(sys.argv[4])
days_since_seed = float(sys.argv[5])
growth_stage = 0  # Default, adjust with sensor data

# Encode plant type
plant_type_enc = 0 if plant_type == "Chilli" else 1

# Compute engineered features
growth_stage_x_days_since_seed = growth_stage * days_since_seed
temperature_x_humidity = temperature_c * humidity_percent
days_since_seed_squared = days_since_seed**2
height_cm_squared = height_cm**2
temperature_f = (temperature_c * 9/5) + 32
temperature_humidity_index = temperature_f - (0.55 * (1 - humidity_percent/100) * (temperature_f - 58))
height_per_day = height_cm / (days_since_seed + 1e-6)  # Avoid division by zero
growth_stage_sqrt_days = growth_stage * np.sqrt(days_since_seed)

# Prepare input data
input_data = torch.FloatTensor([
    growth_stage, plant_type_enc, height_cm, temperature_c, humidity_percent,
    days_since_seed, growth_stage_x_days_since_seed, temperature_x_humidity,
    days_since_seed_squared, height_cm_squared, temperature_humidity_index,
    height_per_day, growth_stage_sqrt_days
])

# Make prediction
with torch.no_grad():
    prediction = model(input_data)
    print(prediction.item())