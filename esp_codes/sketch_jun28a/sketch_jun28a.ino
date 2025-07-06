#include <ESP8266WiFi.h>
#include <PubSubClient.h>  // Ensure you're using the right library for MQTT

// WiFi and MQTT broker credentials
const char* ssid = "rv"; // Router SSID
const char* password = "ravija12345"; // Router password
const char* mqtt_server = "test.mosquitto.org"; // MQTT broker

// Sensor pins
const float humiditySensor = D5;   // Water level sensor connected to D2

// Create WiFi and MQTT clients
WiFiClient espClient;
PubSubClient client(espClient);

// wifi setup
void setup_wifi() {
  delay(10);
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" WiFi connected");
}

// callback for the mqtt topic
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

// wifi reconnect
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP8266Client_" + String(WiFi.macAddress());

    if (client.connect(clientId.c_str())) {
      Serial.println("Connected to MQTT broker");
      client.subscribe("sensor/device");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  
  pinMode(humiditySensor, INPUT_PULLUP);   // Water sensor with internal pull-up resistor
  
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  static unsigned long lastMsg = 0;
  unsigned long now = millis();
  if (now - lastMsg > 2000) {  // Send data every 2 seconds
    lastMsg = now;

    // Read water sensor value (digital)
    float waterLevel = digitalRead(humiditySensor);

    // Prepare JSON payload for all sensors
    String payload = "{\"water_level\": " + String(waterLevel) + "}";

    Serial.print("Attempting to publish: ");
    Serial.println(payload);

    // Publish the data to the MQTT broker
    if (client.publish("sensor/data", payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.println("Message publish failed");
    }
  }
}