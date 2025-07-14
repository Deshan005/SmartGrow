import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const [waterLevel, setWaterLevel] = useState(null);
  const [status, setStatus] = useState('Monitoring...');
  const [height, setHeight] = useState(0);
  const [temperature, setTemperature] = useState(25);
  const [humidity, setHumidity] = useState(60);
  const [growthStage, setGrowthStage] = useState('Unknown');
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [isPumpRunning, setIsPumpRunning] = useState(false);

  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }

    const setAutomaticMode = async () => {
      try {
        const response = await fetch("http://192.168.56.1:3001/set_mode", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ mode: "automatic" }),
        });
        const data = await response.json();
        if (!data.success) {
          console.error("Failed to set automatic mode:", data.error);
        }
      } catch (err) {
        console.error("Error setting automatic mode:", err);
      }
    };
    setAutomaticMode();

    const ws = new WebSocket('ws://192.168.56.1:5001');
    ws.onopen = () => console.log('WebSocket Connected');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data.trim());
        setWaterLevel(data.waterLevel ?? null);
        setStatus(data.status || 'Monitoring...');
        setHeight(data.height || 0);
        setTemperature(data.temperature || 25);
        setHumidity(data.humidity || 0);
        setGrowthStage(data.growth_stage || 'Unknown');
        setIsPumpRunning(data.status === "Dry");
      } catch (err) {
        console.warn('⚠️ Received non-JSON message:', event.data);
      }
    };

    ws.onerror = (error) => console.log('WebSocket Error:', error);
    ws.onclose = () => console.log('WebSocket Disconnected');
    return () => ws.close();
  }, [token, navigate]);

  const handleStopPump = async () => {
    if (!token) return navigate("/login");

    try {
      const response = await fetch("http://192.168.56.1:3001/stop_pump", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      });
      const data = await response.json();
      if (data.success) {
        alert("Pump stopped successfully.");
        setIsPumpRunning(false);
      } else {
        alert("Failed to stop pump: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error stopping pump:", err);
      alert("An error occurred while stopping the pump.");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate("/login");
  };

return (
  <div className="relative min-h-screen flex flex-col bg-black/70">
    <img
      src="https://newvistas.com/wp-content/uploads/2024/04/Greenhouse-1.jpeg"
      alt="greenhouse"
      className="absolute inset-0 w-full h-full object-cover z-0"
    />
    <div className="absolute inset-0 bg-black/60 z-0" />

    <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-10 sm:px-6 sm:py-12">
      
      <div className="bg-white/9 backdrop-blur-md px-6 py-4 rounded-xl shadow-lg text-center mb-10">
        <h1 className="text-3xl sm:text-4xl font-bold text-white">
          SmartGrow Monitoring Dashboard
        </h1>
        <p className="text-white/90 text-sm sm:text-base mt-2">Real-time greenhouse sensor tracking and water pump automation</p>
      </div>

      <div className="bg-white/20 backdrop-blur-md p-6 sm:p-8 rounded-2xl shadow-xl max-w-5xl w-full grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Water Level</h2>
          <p className="text-3xl font-bold text-green-900">
            {waterLevel !== null ? `${waterLevel}%` : 'Loading...'}
          </p>
          <p className="text-gray-700 mt-2">Updated via WebSocket</p>
        </div>

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">System Status</h2>
          <p className="text-2xl font-medium text-green-900">{status}</p>
          <p className="text-gray-700 mt-2">Current soil condition</p>
        </div>

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Humidity</h2>
          <p className="text-2xl font-medium text-green-900">{humidity}%</p>
          <p className="text-gray-700 mt-2">Plant zone humidity</p>
        </div>

        {/* <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Growth Stage</h2>
          <p className="text-2xl font-medium text-green-900">{growthStage}</p>
          <p className="text-gray-700 mt-2">Development stage from model</p>
        </div> */}
      </div>

      <div className="text-center mt-6">
        <div className="flex flex-wrap justify-center gap-4">
          <button
            onClick={() => navigate('/manual-water-set')}
            className="bg-green-700 hover:bg-green-800 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-58"
          >
            Manual Water Set
          </button>
          <button
            onClick={() => navigate('/user-details')}
            className="bg-green-700 hover:bg-green-800 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            User Profile
          </button>
          <button
            onClick={handleStopPump}
            disabled={!isPumpRunning}
            className="bg-red-800 hover:bg-red-900 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            Stop Pump
          </button>
          <button
            onClick={handleLogout}
            className="bg-red-800 hover:bg-red-900 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            Logout
          </button>
        </div>
      </div>

    </div>
  </div>
);
}
