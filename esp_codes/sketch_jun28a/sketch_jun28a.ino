#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ------------------- WiFi & Server Configuration -------------------
const char* ssid = "Galaxy A122A41";
const char* password = "deshan2005";
const char* serverUrl = "http://10.79.128.253:3001";  // Backend base URL

// ------------------- Sensor Configuration -------------------
const int soilMoisturePin = A0;  // Soil moisture sensor
#define DHTPIN D2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);
#define MOTOR_PIN D3
bool motorState = LOW;
unsigned long motorStartTime = 0;
unsigned long pumpDuration = 0;
bool manualPumpActive = false;
String systemMode = "automatic";

// ------------------- WiFi Connection -------------------
void setup_wifi() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected, IP: " + WiFi.localIP().toString());
}

// ------------------- Send Sensor Data to Backend -------------------
void sendSensorData(int moistureValue, String moistureStatus, float humidity) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    // Build JSON payload
    StaticJsonDocument<200> doc;
    doc["moisture"] = moistureValue;
    doc["status"] = moistureStatus;
    doc["humidity"] = humidity;

    String payload;
    serializeJson(doc, payload);
    Serial.println("Payload: " + payload);

    // Send POST to /data
    if (http.begin(client, String(serverUrl) + "/data")) {
      http.addHeader("Content-Type", "application/json");
      int code = http.POST(payload);
      Serial.print("POST code: ");
      Serial.println(code);

      if (code > 0) {
        String resp = http.getString();
        Serial.println("Server reply: " + resp);
      } else {
        Serial.println("POST failed: " + http.errorToString(code));
      }
      http.end();
    } else {
      Serial.println("Unable to connect to server for sending data");
    }
  } else {
    Serial.println("Wi-Fi disconnected");
  }
}

// ------------------- Poll Commands from Backend -------------------
void pollCommands() {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    // Send GET to /get_command
    if (http.begin(client, String(serverUrl) + "/get_command")) {
      int code = http.GET();
      Serial.print("GET code: ");
      Serial.println(code);

      if (code > 0) {
        String payload = http.getString();
        Serial.println("Server reply: " + payload);

        // Parse JSON response
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);
        if (error) {
          Serial.print("JSON parsing failed: ");
          Serial.println(error.c_str());
          return;
        }

        // Update system mode
        systemMode = doc["mode"].as<String>();
        Serial.print("System mode: ");
        Serial.println(systemMode);

        // Handle commands (only in manual mode)
        if (systemMode == "manual") {
          String action = doc["command"]["action"];
          if (action == "set_water_level") {
            String plant_type = doc["command"]["plant_type"];
            String growth_stage = doc["command"]["growth_stage"];
            float water_level = doc["command"]["water_level"];
            pumpDuration = doc["command"]["pump_duration"];

            Serial.println("Received manual water level command:");
            Serial.print("Plant Type: ");
            Serial.println(plant_type);
            Serial.print("Growth Stage: ");
            Serial.println(growth_stage);
            Serial.print("Water Level: ");
            Serial.print(water_level);
            Serial.println("%");
            Serial.print("Pump Duration: ");
            Serial.print(pumpDuration);
            Serial.println("ms");

            // Activate pump
            digitalWrite(MOTOR_PIN, LOW);
            motorState = LOW;
            manualPumpActive = true;
            motorStartTime = millis();
            Serial.println("Pump turned ON for manual watering");
          } else if (action == "stop_pump") {
            digitalWrite(MOTOR_PIN, HIGH);
            motorState = HIGH;
            manualPumpActive = false;
            pumpDuration = 0;
            motorStartTime = 0;
            Serial.println("Pump turned OFF by stop command");
          }
        }
      } else {
        Serial.println("GET failed: " + http.errorToString(code));
      }
      http.end();
    } else {
      Serial.println("Unable to connect to server for polling commands");
    }
  } else {
    Serial.println("Wi-Fi disconnected");
  }
}

// ------------------- Setup -------------------
void setup() {
  Serial.begin(115200);
  setup_wifi();
  dht.begin();

  pinMode(soilMoisturePin, INPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
}

// ------------------- Main Loop -------------------
void loop() {
  // Handle manual pump duration
  if (manualPumpActive && (millis() - motorStartTime >= pumpDuration)) {
    digitalWrite(MOTOR_PIN, LOW);
    motorState = LOW;
    manualPumpActive = false;
    Serial.println("Pump turned OFF after manual watering");
  }

  static unsigned long lastMsg = 0;
  unsigned long now = millis();

  if (now - lastMsg > 3000) {
    lastMsg = now;

    // --- Read Soil Moisture ---
    int moistureValue = analogRead(A0);
    if (moistureValue > 99) {
      // Keep 2 most significant digits
      moistureValue = moistureValue / pow(10, (int)log10(moistureValue) - 1);  
    }
    Serial.println(moistureValue);
    bool isMoistureLow = (moistureValue < 43);
    String moistureStatus = isMoistureLow ? "Dry" : "Wet";

    Serial.print("Soil Moisture: ");
    Serial.print(moistureValue);
    Serial.print(" => Status: ");
    Serial.println(moistureStatus);

    // --- Automatic Motor Control (only in automatic mode) ---
    if (systemMode == "automatic" && !manualPumpActive) {
      if (isMoistureLow) {
        digitalWrite(MOTOR_PIN, LOW);
        motorState = LOW;
        Serial.println("Motor ON - Soil is DRY (automatic mode)");
      } else {
        digitalWrite(MOTOR_PIN, HIGH);
        motorState = HIGH;
        Serial.println("Motor OFF - Soil is WET (automatic mode)");
      }
    } else if (systemMode == "manual") {
      Serial.println("Manual mode active - automatic control disabled");
    }

    // --- Read Humidity ---
    float humidity = dht.readHumidity();
    if (isnan(humidity)) {
      Serial.println("Failed to read from DHT sensor!");
    }
    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    // --- Send Sensor Data and Poll Commands ---
    sendSensorData(moistureValue, moistureStatus, humidity);
    pollCommands();
  }
}