const mqtt = require('mqtt');
const WebSocket = require('ws');

// WebSocket server
const wss = new WebSocket.Server({ host: '0.0.0.0', port: 5001 });

wss.on('connection', (ws) => {
  console.log('Client connected to WebSocket');
  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

function setupMqttHandler(wss) {
  const client = mqtt.connect("mqtt://broker.emqx.io");
  const latestData = {}; // Shared state for latest MQTT data

  client.on("connect", () => {
    console.log("✅ MQTT client connected");
    client.subscribe("sensor/data");
    console.log("Connected to MQTT broker");
  });

  client.on("message", (topic, message) => {
    console.log(`Received message on ${topic}: ${message.toString()}`);

    if (!wss || !wss.clients) {
      console.error("❌ WebSocket server (wss) is not defined correctly.");
      return;
    }

    const data = JSON.parse(message.toString());
    // Add growth stage (simplified assumption based on days_since_seed from dataset)
    data.growth_stage = data.days_since_seed < 30 ? "Seedling" : data.days_since_seed < 60 ? "Growing" : data.days_since_seed < 90 ? "Flowering" : "Fruiting";
    latestData.growth_stage = data.growth_stage;
    latestData.waterLevel = parseInt(data.moisture) > 500 ? 40 : 60; // Simplified water level based on moisture
    latestData.status = data.status;
    latestData.humidity = data.humidity;
    latestData.temperature = data.temperature; // Assuming temperature is sent

    // Broadcast to all connected WebSocket clients
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(latestData));
      }
    });
  });

  client.on("error", (err) => {
    console.error("❌ MQTT Error:", err.message);
  });
}

module.exports = setupMqttHandler;