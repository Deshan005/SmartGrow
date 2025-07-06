import torch
import torch.nn as nn
import sys

class WaterLevelNN(nn.Module):
    def __init__(self):
        super(WaterLevelNN, self).__init__()
        self.layer1 = nn.Linear(6, 12)
        self.layer2 = nn.Linear(12, 6)
        self.output = nn.Linear(6, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

model = WaterLevelNN()
model.load_state_dict(torch.load('ml_scripts/water_level_model.pth'))
model.eval()

plant_type = sys.argv[1]  # "Chilli" or "Brinjal"
height_cm = float(sys.argv[2])
temperature_c = float(sys.argv[3])
humidity_percent = float(sys.argv[4])
days_since_seed = float(sys.argv[5])  # New parameter
growth_stage = 0  # Default, adjust with sensor data

plant_type_enc = 0 if plant_type == "Chilli" else 1

input_data = torch.FloatTensor([growth_stage, plant_type_enc, height_cm, temperature_c, humidity_percent, days_since_seed])

with torch.no_grad():
    prediction = model(input_data)
    print(prediction.item())