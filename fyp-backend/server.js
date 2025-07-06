const express = require('express');
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const axios = require('axios');

const app = express();
app.use(express.json());

// ESP8266 URL (Update with your ESP8266 IP and port)
const ESP8266_URL = 'http://YOUR_ESP8266_IP'; // Example: 'http://192.168.1.100'

// Middleware to verify Firebase ID Token (simplified for demo)
const verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'No token provided' });
  try {
    // In production, use Firebase Admin SDK to verify the token
    jwt.verify(token, 'your-secret-key'); // Placeholder; replace with Firebase public keys
    next();
  } catch (err) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Set water level endpoint
app.post('/setWaterLevel', verifyToken, async (req, res) => {
  try {
    const { level } = req.body;
    await axios.post(`${ESP8266_URL}/setWaterLevel`, { level });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: 'Failed to set water level' });
  }
});

// WebSocket server for real-time sensor updates
const wss = new WebSocket.Server({ port: 5001 });
wss.on('connection', (ws) => {
  console.log('Client connected to WebSocket');
  const interval = setInterval(async () => {
    try {
      const response = await axios.get(`${ESP8266_URL}/sensor`);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(response.data));
      }
    } catch (err) {
      console.error('Error fetching sensor data:', err.message);
    }
  }, 2000);

  ws.on('close', () => {
    console.log('Client disconnected');
    clearInterval(interval);
  });
});

app.listen(5000, () => console.log('Express server running on port 5000'));
console.log('WebSocket server running on port 5001');