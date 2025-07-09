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

    // Broadcast to all connected WebSocket clients
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) { // Corrected from client.OPEN
        client.send(`${message.toString()}`);
      }
    });
  });

  client.on("error", (err) => {
    console.error("❌ MQTT Error:", err.message);
  });
}

module.exports = setupMqttHandler;