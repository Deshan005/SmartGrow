import pandas as pd
import numpy as np

np.random.seed(42)

data = []
for _ in range(400):  # 200 samples per plant type
    plant_type = np.random.choice(["Chilli", "Brinjal"])
    if plant_type == "Chilli":
        stage = np.random.choice(["Seedling", "Growing", "Flowering", "Fruiting"])
        days = np.random.uniform(0, 90)  # 0-90 days from seed
        height = np.random.uniform(5, 70) if stage == "Fruiting" else np.random.uniform(5, 50)
        temp = np.random.uniform(20, 35)
        humidity = np.random.uniform(50, 80)
        water_base = 40  # Base water level
        if stage == "Seedling": water = water_base + np.random.uniform(-5, 5)
        elif stage == "Growing": water = water_base + 5 + np.random.uniform(-5, 5)
        elif stage == "Flowering": water = water_base + 10 + np.random.uniform(-5, 5)
        else: water = water_base + 15 + np.random.uniform(-5, 5)  # Fruiting
    else:  # Brinjal
        stage = np.random.choice(["Seedling", "Growing", "Flowering", "Fruiting"])
        days = np.random.uniform(0, 100)  # 0-100 days from seed
        height = np.random.uniform(5, 80) if stage == "Fruiting" else np.random.uniform(5, 60)
        temp = np.random.uniform(20, 35)
        humidity = np.random.uniform(50, 80)
        water_base = 40
        if stage == "Seedling": water = water_base + np.random.uniform(-5, 5)
        elif stage == "Growing": water = water_base + 5 + np.random.uniform(-5, 5)
        elif stage == "Flowering": water = water_base + 10 + np.random.uniform(-5, 5)
        else: water = water_base + 15 + np.random.uniform(-5, 5)  # Fruiting
    data.append([stage, plant_type, height, temp, humidity, days, max(30, min(70, water))])

df = pd.DataFrame(data, columns=["growth_stage", "plant_type", "height_cm", "temperature_c", "humidity_percent", "days_since_seed", "water_level_percent"])
df.to_csv("ml_scripts/plant_water_dataset.csv", index=False)
print("Dataset generated and saved to ml_scripts/plant_water_dataset.csv")