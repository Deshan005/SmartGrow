# SmartGrow: IoT-Based Greenhouse Smart Agriculture System

SmartGrow is an IoT-driven precision agriculture system designed for greenhouses. It provides real-time environmental monitoring (soil moisture and ambient humidity) and full control over motorized water pumps to ensure optimal plant health. The system replaces manual guesswork with either threshold-based automatic irrigation or an advanced age-based scheduled watering engine.

## 🚀 Features

- **Real-Time Dashboard (React.js):** Sub-second UI updates via WebSocket broadcasting, displaying live soil moisture, humidity, system mode, and plant lifecycle tracking.
- **Hardware Integration (ESP8266):** Robust HTTP REST polling architecture using an ESP8266 with a DHT11 and capacitive soil moisture sensor.
- **Precision Irrigation (Node.js):**
  - **Automatic Mode:** Engages the water pump when soil moisture falls below defined thresholds.
  - **Cron/Schedule Mode:** A daily age-based automation engine that calculates plant growth stages (Seedling, Growing, Flowering, Fruiting) and cross-references a plant-water dataset to deliver exact milliliter volumes.
  - **Manual Mode:** Manual overrides via the dashboard offering dataset-driven calculations for precise watering on demand.
- **Hardware Safety Protocols:** Detects empty tank/low flow events to automatically halt the pump and broadcast an immediate alert to the dashboard.
- **Secure & Multi-User:** Built-in JWT-based authentication and `bcrypt` password hashing. A local MySQL database isolates garden profiles and schedules, supporting multi-user environments.

## 📂 Project Structure

- `fyp-frontend/`: The React.js frontend application.
- `fyp-backend/`: The Node.js/Express.js backend server.
- `esp_codes/`: Arduino C++ `.ino` sketches for the ESP8266 microcontroller.

## 🛠️ Getting Started

### Prerequisites

1.  **Node.js**: Installed on your system.
2.  **MySQL Server**: Running locally.
3.  **Arduino IDE**: Configured with ESP8266 board definitions and the `ArduinoJson` library.

### 1. Database Setup

1.  Open your MySQL terminal or interface (e.g., phpMyAdmin, MySQL Workbench) and create the database:
    ```sql
    CREATE DATABASE smartgrow;
    ```
2.  The backend attempts to connect to the database using the username `root` with no password. If your MySQL server uses a password or a different port (default is 3306), update the connection pool configuration in `fyp-backend/server.js`.

### 2. Running the Backend

1.  Open a terminal inside the `fyp-backend/` directory.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the Node.js server:
    ```bash
    node server.js
    ```
    *The REST API will run on `http://localhost:3001` and the WebSocket server on `ws://localhost:5001`.*

### 3. Running the Frontend

1.  Open a new terminal inside the `fyp-frontend/` directory.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the React application:
    ```bash
    npm start
    ```
    *The dashboard will automatically open at `http://localhost:3000`.*

### 4. Hardware Setup (ESP8266)

1.  Open `esp_codes/sketch_jun28a/sketch_jun28a.ino` in the Arduino IDE.
2.  Update the Wi-Fi credentials:
    ```cpp
    const char* ssid = "YOUR_WIFI_SSID";
    const char* password = "YOUR_WIFI_PASSWORD";
    ```
3.  Update the `serverUrl` string variable to point to your backend machine's local IP address (e.g., `http://192.168.1.100:3001`).
4.  Connect your components:
    - DHT11 Data Pin -> `D2`
    - Capacitive Soil Moisture Sensor Analog Pin -> `A0`
    - 5V Relay Module (Water Pump) -> `D3`
5.  Upload the sketch to the ESP8266. 

## 🛡️ License & Attributions

This project utilizes open-source libraries (MIT and GNU GPL compliant) including ReactJS, Express.js, Tailwind CSS, node-cron, JSONWebToken, and Arduino.
