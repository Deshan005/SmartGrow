const express = require('express');
const WebSocket = require('ws');
const jwt = require('jsonwebtoken');
const axios = require('axios');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const cors = require('cors');
const mqttHandler = require('./mqtt/mqtt');

const app = express();
app.use(express.json());
app.use(cors());

// MySQL Connection
const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '', // Update with your MySQL password if set
  database: 'smartgrow'
});

db.connect((err) => {
  if (err) throw err;
  console.log('Connected to MySQL Database');
  db.query(`
    CREATE TABLE IF NOT EXISTS users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      username VARCHAR(50) UNIQUE NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )`, (err) => {
      if (err) throw err;
      console.log('Users table ready');
    });
});

// ESP8266 URL
const ESP8266_URL = "http://192.168.56.1"; // Corrected to HTTP port
console.log(ESP8266_URL);

// JWT Secret
const JWT_SECRET = 'fyp';

// Middleware to verify JWT
const verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'No token provided' });
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Signup endpoint
app.post('/signup', async (req, res) => {
  const { email, username, password } = req.body;
  if (!email || !username || !password) {
    return res.status(400).json({ error: 'Please fill in all fields.' });
  }

  try {
    const usernameCheck = `SELECT username FROM users WHERE username = ?`;
    db.query(usernameCheck, [username], (err, results) => {
      if (err) return res.status(500).json({ error: err.message });
      if (results.length > 0) return res.status(400).json({ error: 'This username is already taken.' });

      const emailCheck = `SELECT email FROM users WHERE email = ?`;
      db.query(emailCheck, [email], async (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        if (results.length > 0) return res.status(400).json({ error: 'This email is already in use.' });

        const hashedPassword = await bcrypt.hash(password, 10);
        const query = `INSERT INTO users (username, email, password) VALUES (?, ?, ?)`;
        db.query(query, [username, email, hashedPassword], (err, result) => {
          if (err) return res.status(500).json({ error: err.message });
          const token = jwt.sign({ id: result.insertId, username }, JWT_SECRET, { expiresIn: '1h' });
          res.status(201).json({ message: 'User created successfully', token });
        });
      });
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Login endpoint
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const query = `SELECT * FROM users WHERE username = ?`;
  db.query(query, [username], async (err, results) => {
    if (err || results.length === 0) return res.status(400).json({ error: 'Invalid credentials' });
    const user = results[0];
    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(400).json({ error: 'Invalid credentials' });
    const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '1h' });
    res.json({ message: 'Logged in successfully', token });
  });
});

// User details endpoint
app.get('/user-details', verifyToken, (req, res) => {
  const query = `SELECT username, email, created_at FROM users WHERE id = ?`;
  db.query(query, [req.user.id], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    if (results.length === 0) return res.status(404).json({ error: 'User not found' });
    res.json(results[0]);
  });
});

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

// Predict water level endpoint
app.post('/predict_water_level', verifyToken, (req, res) => {
  const { plant_type, height_cm, temperature_c, humidity_percent } = req.body;
  const baseLevel = 40;
  let waterLevel = baseLevel;
  if (plant_type === 'Brinjal') waterLevel += 5;
  if (height_cm > 50) waterLevel += 10;
  if (temperature_c > 30) waterLevel += 5;
  if (humidity_percent < 50) waterLevel += 10;
  res.json({ water_level: Math.min(70, Math.max(30, waterLevel)) });
});

// New endpoint for manual water level setting
app.post('/manual_set_water_level', verifyToken, async (req, res) => {
  const { plant_type, growth_stage } = req.body;
  let waterLevel = 40; // Base level
  if (plant_type === 'Brinjal') waterLevel += 5;
  if (growth_stage === 'Seedling') waterLevel += 10;
  else if (growth_stage === 'Flowering') waterLevel += 5;
  else if (growth_stage === 'Fruiting') waterLevel += 15;
  await axios.post(`${ESP8266_URL}/setWaterLevel`, { level: waterLevel });
  res.json({ success: true, water_level: waterLevel });
});

// WebSocket server with automatic adjustment
const wss = new WebSocket.Server({ port: 5001 });

let latestData = {}; // Store the latest MQTT data

wss.on('connection', (ws) => {
  console.log('Client connected to WebSocket');
  ws.send(JSON.stringify(latestData)); // Send latest data to new client
  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

// Call the MQTT setup function
mqttHandler(wss);

// Automatic water level adjustment for Seedling stage
setInterval(async () => {
  if (latestData.growth_stage === 'Seedling' && latestData.waterLevel) {
    const newLevel = Math.min(70, latestData.waterLevel + 5);
    await axios.post(`${ESP8266_URL}/setWaterLevel`, { level: newLevel });
    latestData.waterLevel = newLevel;
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(latestData));
      }
    });
  }
}, 10000); // Adjust every 10 seconds

app.listen(5000, () => console.log('Express server running on port 5000'));
console.log('WebSocket server running on port 5001');