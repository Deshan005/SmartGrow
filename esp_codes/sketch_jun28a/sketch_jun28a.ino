#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ------------------- WiFi & Server Configuration -------------------
const char* ssid = "Dialog 4G 605";
const char* password = "AAB53B59";
const char* serverUrl = "http://192.168.8.143:3001";  // Backend base URL
// const char* serverUrl = "http://10.79.128.253:3001";  Backend base URL

// ------------------- Sensor Configuration -------------------
const int soilMoisturePin = A0;  // Soil moisture sensor
#define DHTPIN D2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);
#define MOTOR_PIN D3
bool motorState = HIGH;
unsigned long motorStartTime = 0;
unsigned long pumpDuration = 0;
bool manualPumpActive = false;
String systemMode = "standby";
float targetMoisture = 100.0;

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
    StaticJsonDocument<256> doc;
    doc["moisture"] = moistureValue;
    doc["status"] = moistureStatus;
    doc["humidity"] = humidity;
    doc["pumpState"] = (motorState == LOW) ? "ON" : "OFF";

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

        // Parse JSON response - buffer must be large enough for full cron command payload
        StaticJsonDocument<512> doc;
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
          String action = doc["command"]["action"].as<String>();
          String commandIdStr = doc["command"]["id"].as<String>();
          static String lastCommandIdStr = "";

          // EMERGENCY OVERRIDE: Stop pump unconditionally
          if (action == "stop_pump") {
            digitalWrite(MOTOR_PIN, HIGH);
            motorState = HIGH;
            manualPumpActive = false;
            pumpDuration = 0;
            motorStartTime = 0;
            Serial.println("Pump turned OFF by absolute emergency stop command");
          } 
          // Process other commands only once natively
          else if (commandIdStr != "" && commandIdStr != "null" && commandIdStr != lastCommandIdStr) {
            lastCommandIdStr = commandIdStr;

            if (action == "set_water_level") {
              String plant_type = doc["command"]["plant_type"].as<String>();
              String growth_stage = doc["command"]["growth_stage"].as<String>();
              float water_level = doc["command"]["water_level"].as<float>();
              targetMoisture = water_level; // Save for safety stop
              pumpDuration = doc["command"]["pump_duration"].as<unsigned long>();
              Serial.print("Target Moiture Level: ");
              Serial.println(targetMoisture);
              Serial.print("Pump Duration from cron: ");
              Serial.println(pumpDuration);

              // Activate pump
              digitalWrite(MOTOR_PIN, LOW);
              motorState = LOW;
              manualPumpActive = true;
              motorStartTime = millis();
              Serial.println("Pump turned ON for manual watering via tracking ID");
            } else if (action == "start_pump") {
              digitalWrite(MOTOR_PIN, LOW);
              motorState = LOW;
              manualPumpActive = true;
              pumpDuration = 4294967295; // Infinite duration, manually stopped
              motorStartTime = millis();
              Serial.println("Pump turned ON universally by start command");
            }
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
  digitalWrite(MOTOR_PIN, HIGH);
  motorState = HIGH;
}

void loop() {
  static unsigned long lastMsg = 0;
  unsigned long now = millis();

  // --- Global Soil Moisture Reading ---
  int moistureValue = analogRead(A0);
  int currentMoisture = moistureValue;
  if (currentMoisture > 99) {
    currentMoisture = currentMoisture / pow(10, (int)log10(currentMoisture) - 1);
  }

  // Handle manual/automation pump duration and safety threshold
  if (manualPumpActive) {
    // DIAGNOSTIC LOG (every 1 second while pumping)
    static unsigned long lastPumpLog = 0;
    if (now - lastPumpLog > 1000) {
      lastPumpLog = now;
      Serial.print("🛠️ [PUMPING] Current: ");
      Serial.print(currentMoisture);
      Serial.print("% | Target: ");
      Serial.print(targetMoisture);
      Serial.println("%");
    }

    // STOP IF: Timer expires OR Soil moisture passes target
    if ((now - motorStartTime >= pumpDuration) || (currentMoisture >= targetMoisture)) {
      digitalWrite(MOTOR_PIN, HIGH);
      motorState = HIGH;
      manualPumpActive = false;
      Serial.print("🛑 Pump OFF. Trigger: ");
      if (currentMoisture >= targetMoisture) {
         Serial.print("Target Moisture (");
         Serial.print(targetMoisture);
         Serial.println(") reached.");
      } else {
         Serial.println("Watering Duration Timeout.");
      }
    }
  }

  if (now - lastMsg > 3000) {
    lastMsg = now;

    // --- Read Soil Moisture Status ---
    bool isMoistureLow = (currentMoisture < 43);
    String moistureStatus = isMoistureLow ? "Dry" : "Wet";

    Serial.print("Soil Moisture: ");
    Serial.print(currentMoisture);
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
    } else if (systemMode == "standby") {
      Serial.println("Standby mode active - waiting for user command...");
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
    sendSensorData(currentMoisture, moistureStatus, humidity);
    pollCommands();
  }
}