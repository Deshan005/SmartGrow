import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess data
df = pd.read_csv("ml_scripts/plant_water_dataset.csv")
le_stage = LabelEncoder()
le_plant = LabelEncoder()
df["growth_stage"] = le_stage.fit_transform(df["growth_stage"])
df["plant_type"] = le_plant.fit_transform(df["plant_type"])

X = df[["growth_stage", "plant_type", "height_cm", "temperature_c", "humidity_percent", "days_since_seed"]].values
y = df["water_level_percent"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

class WaterLevelNN(nn.Module):
    def __init__(self):
        super(WaterLevelNN, self).__init__()
        self.layer1 = nn.Linear(6, 12)  # 6 inputs (added days_since_seed)
        self.layer2 = nn.Linear(12, 6)
        self.output = nn.Linear(6, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

model = WaterLevelNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss: {test_loss.item()}")

torch.save(model.state_dict(), "ml_scripts/water_level_model.pth")
print("Model trained and saved to ml_scripts/water_level_model.pth")