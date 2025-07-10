import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const [waterLevel, setWaterLevel] = useState(null);
  const [status, setStatus] = useState('Monitoring...');
  const [plantType, setPlantType] = useState('Chilli');
  const [height, setHeight] = useState(0);
  const [temperature, setTemperature] = useState(25);
  const [humidity, setHumidity] = useState(60);
  const [growthStage, setGrowthStage] = useState('Unknown');
  const [motionDetected, setMotionDetected] = useState(null); // Last movement time
  const [movementHistory, setMovementHistory] = useState([]); // Last 5 movement times
  const [showHistory, setShowHistory] = useState(false); // Toggle dropdown
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:5001');

    ws.onopen = () => console.log('WebSocket Connected');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data.trim());
        setWaterLevel(data.waterLevel);
        setStatus(data.status || 'Monitoring...');
        setHeight(data.height || 0);
        setTemperature(data.temperature || 25);
        setHumidity(data.humidity || 60);
        setGrowthStage(data.growth_stage || 'Unknown');
        // Handle motion data
        if (data.motionTime) {
          const newTime = new Date(data.motionTime).toLocaleString();
          setMotionDetected(newTime);
          setMovementHistory(prev => {
            const updatedHistory = [newTime, ...prev].slice(0, 5); // Keep last 5
            return updatedHistory;
          });
        }
      } catch (err) {
        console.warn('⚠️ Received non-JSON message:', event.data);
      }
    };

    ws.onerror = (error) => console.log('WebSocket Error:', error);

    return () => ws.close();
  }, []);

  const autoSetWaterLevel = async () => {
    if (!token) return navigate('/login');
    const response = await fetch('http://localhost:5000/predict_water_level', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        plant_type: plantType,
        height_cm: height,
        temperature_c: temperature,
        humidity_percent: humidity,
      }),
    });
    const data = await response.json();
    if (data.water_level) {
      await fetch('http://localhost:5000/setWaterLevel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ level: data.water_level }),
      });
      setWaterLevel(data.water_level);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate("/login");
  };

  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">FYP Water Level Monitor Dashboard</h1>
        <p className="text-gray-600 mt-2">Real-time monitoring and control</p>
      </div>
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Water Level</h2>
          <p className="text-4xl font-bold text-green-600">
            {waterLevel !== null ? `${waterLevel}%` : 'Loading...'}
          </p>
          <p className="text-gray-500 mt-2">Last updated via WebSocket</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">System Status</h2>
          <p className="text-2xl font-medium text-gray-800">{status}</p>
          <p className="text-gray-500 mt-2">Check sensor connection</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Growth Stage</h2>
          <p className="text-2xl font-medium text-gray-800">{growthStage}</p>
          <p className="text-gray-500 mt-2">Current plant growth stage</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Last Motion Detected</h2>
          <p className="text-2xl font-medium text-gray-800">
            {motionDetected || 'No movement yet'}
          </p>
          <p className="text-gray-500 mt-2 flex items-center">
            Last movement time
            <span
              className="ml-2 cursor-pointer text-blue-600 hover:text-blue-800"
              onClick={toggleHistory}
            >
              ▼q
            </span>
          </p>
          {showHistory && movementHistory.length > 0 && (
            <div className="mt-2 bg-gray-50 p-2 rounded">
              <h3 className="text-md font-semibold text-gray-700">Movement History</h3>
              <ul className="list-disc pl-5">
                {movementHistory.map((time, index) => (
                  <li key={index} className="text-gray-600">{time}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
      <div className="max-w-4xl mx-auto mt-8 text-center">
        <select
          value={plantType}
          onChange={(e) => setPlantType(e.target.value)}
          className="mb-4 p-2 border rounded"
        >
          <option value="Chilli">Chilli</option>
          <option value="Brinjal">Brinjal</option>
        </select>
        <button
          onClick={autoSetWaterLevel}
          className="bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-green-700 transition duration-200"
        >
          Auto Set Water Level
        </button>
        <button
          onClick={() => navigate('/user-details')}
          className="ml-4 bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200"
        >
          View User Details
        </button>
        <button
          onClick={() => navigate('/manual-water-set')}
          className="ml-4 bg-yellow-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-yellow-700 transition duration-200"
        >
          Manual Set Water Level
        </button>
      </div>
      <div className="max-w-4xl mx-auto mt-6 text-center">
        <button
          onClick={handleLogout}
          className="bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200"
        >
          Logout
        </button>
      </div>
    </div>
  );
}