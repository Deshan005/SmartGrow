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
  const [pumpState, setPumpState] = useState('OFF');
  const [espConnected, setEspConnected] = useState('loading');
  const [currentDate, setCurrentDate] = useState("");
  const [nextWateringTime, setNextWateringTime] = useState("");
  const [plantStatuses, setPlantStatuses] = useState([]);
  const hasAlerted = React.useRef(false);

  useEffect(() => {
    const updateDateTime = () => {
      const now = new Date();
      setCurrentDate(now.toLocaleDateString() + ' ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }));
      
      const nextWatering = new Date(now);
      nextWatering.setHours(10, 40, 0, 0);
      
      // If the 13:32 time has already passed today, set it to tomorrow
      if (now >= nextWatering) {
        nextWatering.setDate(nextWatering.getDate() + 1);
      }
      
      setNextWateringTime(nextWatering.toLocaleDateString() + ' 10:40');
    };
    
    updateDateTime();
    const intervalId = setInterval(updateDateTime, 1000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }

    const ws = new WebSocket('ws://localhost:5001');
    ws.onopen = () => console.log('WebSocket Connected');

    let connectionTimeout;
    const resetConnectionTimeout = () => {
      clearTimeout(connectionTimeout);
      setEspConnected(true);
      connectionTimeout = setTimeout(() => {
        setEspConnected(false); // No data received for 10 seconds
      }, 10000);
    };

    // Start initial timeout
    connectionTimeout = setTimeout(() => {
      setEspConnected(false);
    }, 10000);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data.trim());
        setWaterLevel(data.waterLevel ?? null);
        setStatus(data.status || 'Monitoring...');
        setHeight(data.height || 0);
        setTemperature(data.temperature || 25);
        setHumidity(data.humidity || 0);
        setGrowthStage(data.growth_stage || 'Unknown');
        setPumpState(data.pumpState || 'OFF');
        setIsPumpRunning(data.pumpState === "ON");
        resetConnectionTimeout(); // We got regular data!
      } catch (err) {
        console.warn('⚠️ Received non-JSON message:', event.data);
      }
    };

    ws.onerror = (error) => console.log('WebSocket Error:', error);
    ws.onclose = () => console.log('WebSocket Disconnected');
    
    // Fetch Plant Statuses
    const fetchStatuses = async () => {
      try {
        const response = await fetch("http://localhost:3001/plant_status", {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setPlantStatuses(data.statuses || []);
        }
      } catch (err) {
        console.error("Failed to fetch plant statuses:", err);
      }
    };
    fetchStatuses();

    return () => {
      ws.close();
      clearTimeout(connectionTimeout);
    };
  }, [token, navigate]);

  const handleStartPump = async () => {
    if (!token) return navigate("/login");

    try {
      const response = await fetch("http://localhost:3001/start_pump", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      });
      if (response.status === 401) {
        localStorage.removeItem("token");
        navigate("/login");
        return;
      }
      const data = await response.json();
      if (data.success) {
        alert("Emergency Pump started.");
        setIsPumpRunning(true);
      } else {
        alert("Failed to start pump: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error starting pump:", err);
      // Fallback
    }
  };

  const handleStopPump = async () => {
    if (!token) return navigate("/login");

    try {
      const response = await fetch("http://localhost:3001/stop_pump", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      });
      if (response.status === 401) {
        localStorage.removeItem("token");
        navigate("/login");
        return;
      }
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
        
        <div className="mt-6 p-4 bg-green-900/40 rounded-lg inline-block text-left border border-white/20">
          <p className="text-white text-lg font-medium tracking-wide">📅 Current Date: <span className="font-light">{currentDate}</span></p>
          <p className="text-green-300 text-lg font-medium tracking-wide mt-1">💧 Next Auto-Watering: <span className="font-light">{nextWateringTime}</span></p>
        </div>

        {/* Plant Stages Display */}
        {plantStatuses.length > 0 && (
          <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl px-4">
            {plantStatuses.map((p, idx) => (
              <div key={idx} className="bg-white/10 backdrop-blur-md border border-white/20 p-4 rounded-xl text-left">
                <div className="flex justify-between items-center mb-1">
                  <h3 className="text-white font-bold text-lg">{p.type}</h3>
                  <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full uppercase font-bold tracking-tighter">
                    {p.stage}
                  </span>
                </div>
                <p className="text-white/70 text-sm italic">Day {p.days} of Growth</p>
                <div className="w-full bg-white/20 h-1 mt-3 rounded-full overflow-hidden">
                   <div 
                      className="bg-green-400 h-full transition-all duration-500" 
                      style={{ width: `${Math.min(100, (p.days / 120) * 100)}%` }} 
                   />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="bg-white/20 backdrop-blur-md p-6 sm:p-8 rounded-2xl shadow-xl max-w-5xl w-full grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Water Level</h2>
          <p className="text-3xl font-bold text-green-900">
            {espConnected === 'loading' ? 'Loading...' : 
             espConnected === false ? <span className="text-xl text-red-600">ESP Not Connected</span> : 
             waterLevel !== null ? `${waterLevel}%` : 'Loading...'}
          </p>
          <p className="text-gray-700 mt-2">Updated via WebSocket</p>
        </div>

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">System Status</h2>
          <p className={`text-2xl font-medium ${espConnected === false ? 'text-red-600' : 'text-green-900'}`}>
            {espConnected === 'loading' ? 'Loading...' : 
             espConnected === false ? 'Offline' : status}
          </p>
          <p className="text-gray-700 mt-2">Current soil condition</p>
        </div>

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Humidity</h2>
          <p className={`text-2xl font-medium ${espConnected === false ? 'text-red-600' : 'text-green-900'}`}>
            {espConnected === 'loading' ? 'Loading...' : 
             espConnected === false ? 'Offline' : `${humidity}%`}
          </p>
          <p className="text-gray-700 mt-2">Plant zone humidity</p>
        </div>

        <div className="bg-white/60 p-4 sm:p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Pump Status</h2>
          <p className={`text-2xl font-medium ${espConnected === false ? 'text-red-600' : pumpState === 'ON' ? 'text-blue-600' : 'text-gray-600'}`}>
            {espConnected === 'loading' ? 'Loading...' : 
             espConnected === false ? 'Offline' : pumpState}
          </p>
          <p className="text-gray-700 mt-2">Current water motor state</p>
        </div>
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
            onClick={() => navigate('/automation-settings')}
            className="bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-58"
          >
            Automation Setup
          </button>
          <button
            onClick={() => navigate('/user-details')}
            className="bg-green-700 hover:bg-green-800 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            User Profile
          </button>
          <button
            onClick={handleStartPump}
            disabled={isPumpRunning}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            Start Pump
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
            className="bg-gray-800 hover:bg-gray-900 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md transition w-48"
          >
            Logout
          </button>
        </div>
      </div>

    </div>
  </div>
);
}
