import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Load and preprocess data
df = pd.read_csv("ml_scripts/plant_water_dataset.csv")

# Encode categorical features
le_stage = LabelEncoder()
le_plant = LabelEncoder()
df["growth_stage"] = le_stage.fit_transform(df["growth_stage"])
df["plant_type"] = le_plant.fit_transform(df["plant_type"])

# Advanced Feature Engineering
df['growth_stage_x_days_since_seed'] = df['growth_stage'] * df['days_since_seed']
df['temperature_x_humidity'] = df['temperature_c'] * df['humidity_percent']
df['days_since_seed_squared'] = df['days_since_seed']**2
df['height_cm_squared'] = df['height_cm']**2
df['temperature_f'] = (df['temperature_c'] * 9/5) + 32  # Convert to Fahrenheit
df['temperature_humidity_index'] = df['temperature_f'] - (0.55 * (1 - df['humidity_percent']/100) * (df['temperature_f'] - 58))
df['height_per_day'] = df['height_cm'] / (df['days_since_seed'] + 1e-6)  # Avoid division by zero
df['growth_stage_sqrt_days'] = df['growth_stage'] * np.sqrt(df['days_since_seed'])

# Update feature set
X = df[["growth_stage", "plant_type", "height_cm", "temperature_c", "humidity_percent",
        "days_since_seed", "growth_stage_x_days_since_seed", "temperature_x_humidity",
        "days_since_seed_squared", "height_cm_squared", "temperature_humidity_index",
        "height_per_day", "growth_stage_sqrt_days"]].values
y = df["water_level_percent"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Define deeper neural network
class DeeperWaterLevelNN(nn.Module):
    def __init__(self, input_features, dropout_rate=0.0):
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

# Instantiate model with best hyperparameters
input_dim = X_train.shape[1]
model = DeeperWaterLevelNN(input_features=input_dim, dropout_rate=0.0)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

# Train the model
epochs = 2000
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "ml_scripts/water_level_model.pth")
print("Model trained and saved to ml_scripts/water_level_model.pth")