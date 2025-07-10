#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>  // For humidity sensor

// ------------------- WiFi & MQTT Configuration -------------------
const char* ssid = "Galaxy A122A41";
const char* password = "deshan2005";
const char* mqtt_server = "broker.emqx.io";

// ------------------- Sensor Configuration -------------------
const int soilMoisturePin = A0;  // Soil moisture sensor
#define DHTPIN D1
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define PIR_PIN D2
#define MOTOR_PIN D3
bool motorState = LOW;

// ------------------- MQTT Setup -------------------
WiFiClient espClient;
PubSubClient client(espClient);

// Motion detection debounce
unsigned long lastMotionTime = 0;
const unsigned long debounceDelay = 5000;

// ------------------- WiFi Connection -------------------
void setup_wifi() {
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.println("MAC: " + WiFi.macAddress());
}

// ------------------- MQTT Callback -------------------
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  for (unsigned int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

// ------------------- Reconnect to MQTT -------------------
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP8266-" + WiFi.macAddress();

    if (client.connect(clientId.c_str())) {
      Serial.println("Connected to MQTT broker");
      client.subscribe("sensor/device");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" — trying again in 5 seconds");
      delay(5000);
    }
  }
}

// ------------------- Setup -------------------
void setup() {
  Serial.begin(115200);
  setup_wifi();
  dht.begin();

  pinMode(soilMoisturePin, INPUT);
  pinMode(PIR_PIN, INPUT_PULLUP);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

// ------------------- Main Loop -------------------
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  static unsigned long lastMsg = 0;
  unsigned long now = millis();

  if (now - lastMsg > 3000) {
    lastMsg = now;

    // --- Read Soil Moisture ---
    int moistureValue = analogRead(soilMoisturePin);
    bool isMoistureLow = (moistureValue < 200);  // Adjust threshold as needed
    String moistureStatus = isMoistureLow ? "Dry" : "Wet";

    Serial.print("Soil Moisture: ");
    Serial.print(moistureValue);
    Serial.print(" => Status: ");
    Serial.println(moistureStatus);

    // --- Motor Control ---
    if (moistureStatus == "Wet" ) {
      digitalWrite(MOTOR_PIN, HIGH);  // Turn motor ON
      Serial.println("Motor OFF - Soil is WET");;
    } else {
      digitalWrite(MOTOR_PIN, LOW);  // Turn motor OFF
      Serial.println("Motor ON - Soil is DRY");
    }

    // --- Read Humidity ---
    float humidity = dht.readHumidity();
    if (isnan(humidity)) {
      Serial.println("Failed to read from DHT sensor!");
      return;
    }
    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    // --- Read Motion ---
    int motionDetected = digitalRead(PIR_PIN);
    String motionStatus = (motionDetected == HIGH) ? "Motion Detected" : "No Motion";
    unsigned long currentMotionTime = 0;

    if (motionDetected == HIGH && (now - lastMotionTime > debounceDelay)) {
      lastMotionTime = now;
      currentMotionTime = lastMotionTime;
      Serial.println("Motion detected at: " + String(currentMotionTime));
    }

    // --- Create MQTT JSON Payload ---
    String payload = "{";
    payload += "\"moisture\": " + String(moistureValue) + ",";
    payload += "\"status\": \"" + moistureStatus + "\",";
    payload += "\"humidity\": " + String(humidity) + ",";
    payload += "\"motion\": \"" + motionStatus + "\"";

    if (currentMotionTime > 0) {
      payload += ",\"motionTime\": " + String(currentMotionTime);
    }

    payload += "}";

    Serial.print("Publishing payload: ");
    Serial.println(payload);

    if (client.publish("sensor/data", payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.println("Message publish failed");
    }
  }
}
