// const WebSocket = require('ws');

// // WebSocket server for real-time sensor updates
// const wss = new WebSocket.Server({ port: 5001 });
// wss.on('connection', (ws) => {
//   console.log('Client connected to WebSocket');
//   const interval = setInterval(async () => {
//     try {
//       const response = await axios.get(`${ESP8266_URL}/sensor`);
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify(response.data));
//       }
//     } catch (err) {
//       console.error('Error fetching sensor data:', err.message);
//     }
//   }, 2000);

//   ws.on('close', () => {
//     console.log('Client disconnected');
//     clearInterval(interval);
//   });
// });

// console.log('WebSocket server running on port 5001');

// module.exports = wss;