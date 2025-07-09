#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>  // For humidity sensor

// ------------------- WiFi & MQTT Configuration -------------------
const char* ssid = "rv";                     // WiFi SSID
const char* password = "ravija12345";        // WiFi Password
const char* mqtt_server = "broker.emqx.io";  // MQTT Broker address

// ------------------- Sensor Configuration -------------------
// Soil Moisture Sensor
const int soilMoisturePin = A0;  // Analog pin A0

// DHT Sensor (Humidity & Temperature)
#define DHTPIN D1      // Connect DHT sensor data pin to D6
#define DHTTYPE DHT11  // Or use DHT22 if using that sensor
DHT dht(DHTPIN, DHTTYPE);

// Motion Sensor (motion)
#define PIR_PIN D2


// ------------------- MQTT Setup -------------------
WiFiClient espClient;
PubSubClient client(espClient);

// ------------------- WiFi Connection -------------------
void setup_wifi() {
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  // Wait for connection
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
      client.subscribe("sensor/device");  // Subscribe to topic if needed
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

  setup_wifi();                     // Connect to WiFi
  dht.begin();                      // Start DHT sensor
  pinMode(soilMoisturePin, INPUT);  // Setup soil pin
  pinMode(PIR_PIN, INPUT_PULLUP);   // motion

  client.setServer(mqtt_server, 1883);  // Set MQTT broker
  client.setCallback(callback);         // Set callback for incoming MQTT
}

// ------------------- Main Loop -------------------
void loop() {
  if (!client.connected()) {
    reconnect();  // Ensure MQTT connection
  }
  client.loop();

  static unsigned long lastMsg = 0;
  unsigned long now = millis();

  if (now - lastMsg > 3000) {  // Send every 3 seconds
    lastMsg = now;

    // ----------- Read Soil Moisture -----------
    int moistureValue = analogRead(soilMoisturePin);
    Serial.print("Analog Moisture Value: ");
    Serial.println(moistureValue);

    String moistureStatus = (moistureValue < 500) ? "Water Available" : "Water Not Available";

    // ----------- Read Humidity -----------
    float humidity = dht.readHumidity();

    if (isnan(humidity)) {
      Serial.println("Failed to read from DHT sensor!");
      return;
    }

    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    // Motion Sensor
    int motionDetected = digitalRead(PIR_PIN);
    String motionStatus = (motionDetected == HIGH) ? "Motion Detected" : "No Motion";

    Serial.println(motionStatus);


    // ----------- Create JSON Payload -----------
    String payload = "{";
    payload += "\"moisture\": " + String(moistureValue) + ",";
    payload += "\"status\": \"" + moistureStatus + "\",";
    payload += "\"humidity\": " + String(humidity) + ",";
    payload += "\"motion\": \"" + motionStatus + "\"";
    payload += "}";


    Serial.print("Publishing payload: ");
    Serial.println(payload);

    // ----------- Publish to MQTT -----------
    if (client.publish("sensor/data", payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.println("Message publish failed");
    }
  }
}
