const express = require('express');
const jwt = require('jsonwebtoken');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const cors = require('cors');
const WebSocket = require('ws');
const cron = require('node-cron');
const plantWaterDataset = require('./plantWaterDataset');

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

// HEARTBEAT: Log every 1 minute to confirm server is alive and confirm system time
cron.schedule('* * * * *', () => {
   console.log(`💓 [HEARTBEAT] Server active. 24h Time: ${new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}`);
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
      
      const createPlantsTableQuery = `
        CREATE TABLE IF NOT EXISTS user_plants (
          id INT AUTO_INCREMENT PRIMARY KEY,
          user_id INT NOT NULL,
          plant_type VARCHAR(50) NOT NULL,
          seed_date DATE NOT NULL,
          UNIQUE KEY unique_user_plant (user_id, plant_type),
          FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
      `;
      db.query(createPlantsTableQuery, (err) => {
        if (err) console.error('Failed to create user_plants table:', err.message);
        else console.log('User plants table ready');
      });
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
let systemMode = 'standby';

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
  if (!['automatic', 'manual', 'standby'].includes(mode)) {
    return res.status(400).json({ error: 'Invalid mode. Use "automatic", "manual", or "standby".' });
  }
  systemMode = mode;
  console.log(`System mode set to: ${mode}`);
  if (mode === 'automatic' || mode === 'standby') {
    latestCommand = null; // Clear any manual command when switching to non-manual mode
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

// Add or Update a user's plant tracking
app.post('/user_plant', verifyToken, (req, res) => {
  const { plant_type, seed_date } = req.body;
  if (!plant_type || !seed_date) return res.status(400).json({ error: 'Missing plant_type or seed_date' });

  const query = "INSERT INTO user_plants (user_id, plant_type, seed_date) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE seed_date = ?";
  db.query(query, [req.user.id, plant_type, seed_date, seed_date], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ success: true, message: 'Crop tracking started.' });
  });
});

// Get user's tracked plants
app.get('/user_plants', verifyToken, (req, res) => {
  const query = "SELECT plant_type, seed_date FROM user_plants WHERE user_id = ?";
  db.query(query, [req.user.id], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ plants: results });
  });
});

// NEW: Get current status of all plants (Stage, Days, etc.)
app.get('/plant_status', verifyToken, (req, res) => {
  const query = "SELECT plant_type, seed_date FROM user_plants WHERE user_id = ?";
  db.query(query, [req.user.id], (err, results) => {
    if (err) return res.status(500).json({ error: 'Database error' });

    const statuses = results.map(plant => {
      const seedDate = new Date(plant.seed_date);
      const now = new Date();
      const diffTime = Math.abs(now - seedDate);
      const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
      
      let stage = "Fruiting";
      if (diffDays <= 30) stage = "Seedling";
      else if (diffDays <= 60) stage = "Growing";
      else if (diffDays <= 90) stage = "Flowering";

      return {
        type: plant.plant_type,
        days: diffDays,
        stage: stage,
        seedDate: plant.seed_date
      };
    });

    res.json({ success: true, statuses });
  });
});

// Sensor data endpoint (log to terminal only)
app.post('/data', (req, res) => {
  const { moisture, status, humidity, pumpState } = req.body;
  console.log('Received sensor data:');
  console.log(`  Moisture: ${moisture}`);
  console.log(`  Status: ${status}`);
  console.log(`  Humidity: ${humidity}`);
  console.log(`  PumpState: ${pumpState}`);

  // Map ESP data to Home.js expected format
  const wsData = {
    waterLevel: moisture, // Map moisture to waterLevel
    status: status,
    humidity: humidity || null,
    pumpState: pumpState || 'OFF',
    height: 0, 
    temperature: 25, 
    growth_stage: 'Unknown'
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
  const { plant_type, growth_stage, water_level, pump_duration_ms } = req.body;
  if (!plant_type || !growth_stage || !water_level) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  try {
    systemMode = 'manual'; // Switch to manual mode
    const cmdId = Math.floor(Math.random() * 1000000);
    latestCommand = {
      id: cmdId,
      action: 'set_water_level',
      plant_type,
      growth_stage,
      // For manual mode, we ignore soil sensor safety to pump EXACT duration
      water_level: 101, 
      pump_duration: pump_duration_ms || 5000
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
    const cmdId = Math.floor(Math.random() * 1000000);
    latestCommand = { id: cmdId, action: 'stop_pump' };
    res.json({ success: true, message: 'Pump stop command sent' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to stop pump' });
  }
});

// Start pump endpoint
app.post('/start_pump', verifyToken, async (req, res) => {
  try {
    systemMode = 'manual'; // Ensure manual mode
    const cmdId = Math.floor(Math.random() * 1000000);
    latestCommand = { id: cmdId, action: 'start_pump' };
    res.json({ success: true, message: 'Pump start command sent' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to start pump' });
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

// Automated Daily Cron Job to set water level based on growth stage
cron.schedule('40 10 * * *', () => {
   console.log("⏰ [CRON] AUTOMATION TRIGGERED! Calculating multi-crop requirements...");

   const query = "SELECT plant_type, seed_date FROM user_plants"; 
   db.query(query, (err, results) => {
     if (err) {
        console.error("❌ [CRON] Database error:", err);
        return;
     }

     if (results.length === 0) {
        console.log("⚠️ [CRON] No plants found in database. Skipping.");
        return;
     }

     let totalWaterLevelNeeded = 0;
     let plantsProcessed = 0;

     results.forEach((userPlant) => {
        // Find Stage dynamically based on age
        const seedDate = new Date(userPlant.seed_date);
        const now = new Date();
        const diffTime = Math.abs(now - seedDate);
        const currentDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

        let currentStage = "Fruiting"; 
        if (currentDays <= 30) currentStage = "Seedling";
        else if (currentDays <= 60) currentStage = "Growing";
        else if (currentDays <= 90) currentStage = "Flowering";

        // Query AI Dataset for this specific stage and type
        const matchingRows = plantWaterDataset.filter(
           (row) => row.plant_type.toLowerCase() === userPlant.plant_type.toLowerCase() &&
                    row.growth_stage.toLowerCase() === currentStage.toLowerCase()
        );

        if (matchingRows.length > 0) {
           const avgWaterLevel = matchingRows.reduce((sum, row) => sum + row.water_level_percent, 0) / matchingRows.length;
           // Ensure it increases with growth: we calculate the requirement organically from the dataset
           totalWaterLevelNeeded += avgWaterLevel;
           plantsProcessed++;
           console.log(`✅ [CRON] ${userPlant.plant_type}: Day ${currentDays} (${currentStage}) | Target Moiture: ${avgWaterLevel.toFixed(2)}%`);
        } else {
           console.log(`⚠️ [CRON] No dataset info for ${userPlant.plant_type} at ${currentStage} stage.`);
        }
     });

     if (plantsProcessed > 0) {
        const finalWaterLevel = totalWaterLevelNeeded / plantsProcessed;
        const SENSOR_TIMEOUT_MS = 60000; // 1 min total window
        const cmdId = Math.floor(Math.random() * 1000000);
        
        systemMode = 'manual';
        latestCommand = {
           id: cmdId,
           action: 'set_water_level',
           plant_type: 'Multi-Crop',
           growth_stage: 'Auto-Trigger',
           water_level: parseFloat(finalWaterLevel.toFixed(2)),
           pump_duration: SENSOR_TIMEOUT_MS 
        };

        console.log(`🚀 [CRON] Command Sent! Target: ${finalWaterLevel.toFixed(2)}% Moisture. (Safety Timeout: 10m)`);
     } else {
        console.log("❌ [CRON] Failed to calculate needs. No command queued.");
     }
   });
}, {
   scheduled: true,
   timezone: "Asia/Colombo"
});

app.listen(3001, () => console.log('Express server running on port 3001'));
