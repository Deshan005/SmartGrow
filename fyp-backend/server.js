const express = require('express');
const jwt = require('jsonwebtoken');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const cors = require('cors');
const WebSocket = require('ws');

const app = express();
app.use(express.json());
app.use(cors());

// MySQL Connection (for users table only)
const db = mysql.createConnection({
 host: 'localhost',
 user: 'root',
 password: '', // Update with your MySQL password
 database: 'smartgrow'
});

db.connect((err) => {
  if (err) {
    console.error('Database connection failed:', err.message);
    return;
  }
  console.log('Connected to MySQL Database');

  const createUsersTableQuery = `
    CREATE TABLE IF NOT EXISTS users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      username VARCHAR(50) UNIQUE NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `;

  db.query(createUsersTableQuery, (err) => {
    if (err) {
      console.error('Failed to create users table:', err.message);
    } else {
      console.log('Users table ready');
    }
  });
});


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

// Store the latest command and mode for ESP8266
let latestCommand = null;
let systemMode = 'automatic'; // Default to automatic mode

// Create WebSocket server on port 5001
const wss = new WebSocket.Server({ port: 5001 });
console.log('WebSocket server running on ws://localhost:5001');

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Broadcast function to send data to all connected WebSocket clients
function broadcastSensorData(data) {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });
}

// Set mode endpoint
app.post('/set_mode', verifyToken, (req, res) => {
  const { mode } = req.body;
  if (!['automatic', 'manual'].includes(mode)) {
    return res.status(400).json({ error: 'Invalid mode. Use "automatic" or "manual".' });
  }
  systemMode = mode;
  console.log(`System mode set to: ${mode}`);
  if (mode === 'automatic') {
    latestCommand = null; // Clear any manual command when switching to automatic
  }
  res.json({ success: true, mode });
});

// Signup endpoint
app.post('/signup', async (req, res) => {
  const { email, username, password } = req.body;
  if (!email || !username || !password) {
    return res.status(400).json({ error: 'Please fill in all fields.' });
  }
  try {
    const usernameCheck = "SELECT username FROM users WHERE username = ?";
    db.query(usernameCheck, [username], (err, results) => {
      if (err) return res.status(500).json({ error: err.message });
      if (results.length > 0) return res.status(400).json({ error: 'This username is already taken.' });
      const emailCheck = "SELECT email FROM users WHERE email = ?";
      db.query(emailCheck, [email], async (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        if (results.length > 0) return res.status(400).json({ error: 'This email is already in use.' });
        const hashedPassword = await bcrypt.hash(password, 10);
        const query = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)";
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
  const query = "SELECT * FROM users WHERE username = ?";
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
  const query = "SELECT username, email, created_at FROM users WHERE id = ?";
  db.query(query, [req.user.id], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    if (results.length === 0) return res.status(404).json({ error: 'User not found' });
    res.json(results[0]);
  });
});

// Sensor data endpoint (log to terminal only)
app.post('/data', (req, res) => {
  const { moisture, status, humidity } = req.body;
  console.log('Received sensor data:');
  console.log(`  Moisture: ${moisture}`);
  console.log(`  Status: ${status}`);
  console.log(`  Humidity: ${humidity}`);

  // Map ESP data to Home.js expected format
  const wsData = {
    waterLevel: moisture, // Map moisture to waterLevel
    status: status,
    humidity: humidity || null,
    height: 0, // Placeholder (not provided by ESP)
    temperature: 25, // Placeholder (not provided by ESP)
    growth_stage: 'Unknown' // Placeholder (not provided by ESP)
  };

  // Broadcast to WebSocket clients
  broadcastSensorData(wsData);

  res.json({ success: true });
});

// Get command endpoint for ESP8266
app.get('/get_command', (req, res) => {
  res.json({
    mode: systemMode,
    command: latestCommand || { action: 'none' }
  });
});

// Manual water level endpoint
app.post('/manual_set_water_level', verifyToken, async (req, res) => {
  const { plant_type, growth_stage, water_level } = req.body;
  if (!plant_type || !growth_stage || !water_level) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  try {
    systemMode = 'manual'; // Switch to manual mode
    const pumpDuration = Math.round(water_level * 100); // Convert to ms
    latestCommand = {
      action: 'set_water_level',
      plant_type,
      growth_stage,
      water_level: parseFloat(water_level),
      pump_duration: pumpDuration
    };
    res.json({ success: true, water_level });
  } catch (err) {
    res.status(500).json({ error: 'Failed to set water level' });
  }
});

// Stop pump endpoint
app.post('/stop_pump', verifyToken, async (req, res) => {
  try {
    systemMode = 'manual'; // Ensure manual mode
    latestCommand = { action: 'stop_pump' };
    res.json({ success: true, message: 'Pump stop command sent' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to stop pump' });
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

app.listen(3001, () => console.log('Express server running on port 3001'));