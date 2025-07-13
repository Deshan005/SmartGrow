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
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [isPumpRunning, setIsPumpRunning] = useState(false);

  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }

    // Set automatic mode on mount
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
        setWaterLevel(data.waterLevel !== undefined ? data.waterLevel : null);
        setStatus(data.status || 'Monitoring...');
        setHeight(data.height || 0);
        setTemperature(data.temperature || 25);
        setHumidity(data.humidity || 0);
        setGrowthStage(data.growth_stage || 'Unknown');
        setIsPumpRunning(data.status === "Dry"); // Assume pump is running when status is "Dry"
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
    <>
      <style>
        {`
          .responsive-bg {
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            width: 100vw;
            position: relative;
          }
          @media (max-width: 640px) {
            .responsive-bg {
              background-size: contain;
              background-attachment: scroll;
            }
            .responsive-card {
              padding: 1rem;
            }
            .responsive-title {
              font-size: 1.5rem;
            }
            .responsive-text {
              font-size: 0.875rem;
            }
            .responsive-button {
              padding: 0.5rem 1rem;
              font-size: 0.875rem;
            }
          }
          @media (min-width: 641px) and (max-width: 1024px) {
            .responsive-bg {
              background-size: cover;
            }
            .responsive-card {
              padding: 1.5rem;
            }
            .responsive-title {
              font-size: 2rem;
            }
            .responsive-button {
              padding: 0.75rem 1.5rem;
              font-size: 1rem;
            }
          }
        `}
      </style>
      <div
        className="responsive-bg p-4 sm:p-6 md:p-8"
        style={{
          backgroundImage: `url(https://newvistas.com/wp-content/uploads/2024/04/Greenhouse-1.jpeg)`,
        }}
      >
        <div className="text-center mb-6 sm:mb-8">
          <h1 className="responsive-title text-2xl sm:text-3xl font-bold text-white bg-black/50 p-2 rounded-lg">
            FYP Water Level Monitor Dashboard
          </h1>
          <p className="responsive-text text-white mt-2 bg-black/50 p-1 rounded">
            Real-time monitoring and control
          </p>
        </div>
        <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
          <div className="responsive-card bg-green-600/80 p-4 sm:p-6 rounded-lg shadow-md">
            <h2 className="text-lg sm:text-xl font-semibold text-black mb-2">Water Level</h2>
            <p className="text-3xl sm:text-4xl font-bold text-black">
              {waterLevel !== null ? `${waterLevel}%` : 'Loading...'}
            </p>
            <p className="text-black/80 mt-2">Last updated via WebSocket</p>
          </div>

          <div className="responsive-card bg-green-600/80 p-4 sm:p-6 rounded-lg shadow-md">
            <h2 className="text-lg sm:text-xl font-semibold text-black mb-2">System Status</h2>
            <p className="text-xl sm:text-2xl font-medium text-black">{status}</p>
            <p className="text-black/80 mt-2">Check sensor connection</p>
          </div>

          {/* Uncomment and style as needed
          <div className="responsive-card bg-green-500/80 p-4 sm:p-6 rounded-lg shadow-md">
            <h2 className="text-lg sm:text-xl font-semibold text-white mb-2">Growth Stage</h2>
            <p className="text-xl sm:text-2xl font-medium text-white">{growthStage}</p>
            <p className="text-white/80 mt-2">Current plant growth stage</p>
          </div>
          */}

          <div className="responsive-card bg-green-600/80 p-4 sm:p-6 rounded-lg shadow-md">
            <h2 className="text-lg sm:text-xl font-semibold text-black mb-2">Humidity Level</h2>
            <p className="text-xl sm:text-2xl font-medium text-black">{humidity}%</p>
            <p className="text-black/80 mt-2">Current plant humidity level</p>
          </div>
        </div>
        <div className="max-w-4xl mx-auto mt-6 sm:mt-8 text-center">
          <button
            onClick={() => navigate('/manual-water-set')}
            className="responsive-button ml-2 sm:ml-4 bg-yellow-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-base sm:text-lg font-semibold shadow-md hover:bg-yellow-700 transition duration-200"
          >
            Manual Set Water Level
          </button>
          <button
            onClick={() => navigate('/user-details')}
            className="responsive-button ml-2 sm:ml-4 bg-blue-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-base sm:text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200"
          >
            View User Details
          </button>
          <button
            onClick={handleStopPump}
            className="responsive-button ml-2 sm:ml-4 bg-red-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-base sm:text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200"
            disabled={!isPumpRunning}
          >
            Stop Pump
          </button>
        </div>
        <div className="max-w-4xl mx-auto mt-4 sm:mt-6 text-center">
          <button
            onClick={handleLogout}
            className="responsive-button bg-red-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-base sm:text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200"
          >
            Logout
          </button>
        </div>
      </div>
    </>
  );
}