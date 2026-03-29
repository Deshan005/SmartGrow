import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { plantWaterDataset } from "./plantWaterDataset";

export default function ManualWaterSet() {
  const navigate = useNavigate();
  const [plantType, setPlantType] = useState("Select");
  const [growthStage, setGrowthStage] = useState("Select");
  const [token] = useState(localStorage.getItem("token"));
  const [dataset, setDataset] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isPumpRunning, setIsPumpRunning] = useState(false);
  const [waterLevelDisplay, setWaterLevelDisplay] = useState("Select crop and growth stage");
  const [waterAmount, setWaterAmount] = useState(null); 
  const [pumpTime, setPumpTime] = useState(null); 
  const [isSubmitting, setIsSubmitting] = useState(false); 
  const [pumpStartTime, setPumpStartTime] = useState(null); 
  const [espConnected, setEspConnected] = useState('loading');

  // Constants
  const FLOW_RATE = 2500; // mL per minute
  const MAX_VOLUME = 500; // 100% water level = 500 mL (0.5 liter)

  // Set manual mode on component mount
  useEffect(() => {
    const setManualMode = async () => {
      if (!token) return navigate("/login");
      try {
        const response = await fetch("http://localhost:3001/set_mode", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ mode: "manual" }),
        });
        if (response.status === 401) {
          localStorage.removeItem("token");
          navigate("/login");
          return;
        }
        const data = await response.json();
        if (!data.success) {
          console.error("Failed to set manual mode:", data.error);
          alert("Failed to set manual mode: " + (data.error || "Unknown error"));
        }
      } catch (err) {
        console.error("Error setting manual mode:", err);
        alert("Error setting manual mode: Server unreachable");
      }
    };
    setManualMode();

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
        if (data.pumpState) {
          setIsPumpRunning(data.pumpState === "ON");
        }
        resetConnectionTimeout(); // We got regular data!
      } catch (err) {
        console.warn('⚠️ Received non-JSON message:', event.data);
      }
    };

    ws.onerror = (error) => console.log('WebSocket Error:', error);
    ws.onclose = () => console.log('WebSocket Disconnected');
    
    return () => {
      ws.close();
      clearTimeout(connectionTimeout);
    };

  }, [token, navigate]);

  // Clean and load dataset
  useEffect(() => {
    const cleanedData = plantWaterDataset.map((row) => ({
      growth_stage: row.growth_stage ? row.growth_stage.trim().toLowerCase() : "",
      plant_type: row.plant_type ? row.plant_type.trim().toLowerCase() : "",
      water_level_percent: row.water_level_percent ? parseFloat(row.water_level_percent) || 0 : 0,
    }));
    setDataset(cleanedData);
    setLoading(false);
  }, []);

  // Calculate water level, amount, and pump time when selections change
  useEffect(() => {
    if (plantType === "Select" || growthStage === "Select") {
      setWaterLevelDisplay("Select crop and growth stage");
      setWaterAmount(null);
      setPumpTime(null);
      return;
    }

    const waterLevel = findWaterLevel();
    if (waterLevel === 0) {
      setWaterLevelDisplay("No data available");
      setWaterAmount(null);
      setPumpTime(null);
    } else {
      setWaterLevelDisplay(`${waterLevel}%`);
      const waterMilliliters = (waterLevel / 100) * MAX_VOLUME; // Convert percentage to mL
      const timeMinutes = waterMilliliters / FLOW_RATE; // Time in minutes
      const timeSeconds = timeMinutes * 60*2; // Convert to seconds
      setWaterAmount(waterMilliliters.toFixed(2));
      setPumpTime(timeSeconds.toFixed(2));
    }
  }, [plantType, growthStage, dataset]);

  // Automatically stop pump after pumpTime duration
  useEffect(() => {
    let timer;
    if (isPumpRunning && pumpStartTime && pumpTime) {
      const elapsedTimeMs = Date.now() - pumpStartTime;
      const pumpTimeMs = pumpTime * 1000; // Convert pumpTime to milliseconds
      if (elapsedTimeMs >= pumpTimeMs) {
        handleStopPump();
      } else {
        timer = setTimeout(() => {
          handleStopPump();
        }, pumpTimeMs - elapsedTimeMs);
      }
    }
    return () => clearTimeout(timer);
  }, [isPumpRunning, pumpStartTime, pumpTime]);

  // Average water level for selected plant type and growth stage
  const findWaterLevel = () => {
    if (plantType === "Select" || growthStage === "Select") return 0;
    const matchingRows = dataset.filter(
      (row) =>
        row.plant_type === plantType.trim().toLowerCase() &&
        row.growth_stage === growthStage.trim().toLowerCase()
    );
    if (matchingRows.length === 0) return 0;
    // Calculate average water level for consistency
    const avgWaterLevel = matchingRows.reduce((sum, row) => sum + row.water_level_percent, 0) / matchingRows.length;
    return avgWaterLevel.toFixed(2);
  };

  // Handle setting water level and activating pump
  const handleSetWaterLevel = async () => {
    if (!token) return navigate("/login");
    if (plantType === "Select" || growthStage === "Select") {
      alert("Please select a crop and growth stage");
      return;
    }
    if (waterLevelDisplay === "No data available") {
      alert("No water level data available for this selection");
      return;
    }

    try {
      setIsSubmitting(true);
      const waterLevel = findWaterLevel();
      const pumpDurationMs = (pumpTime * 1000).toFixed(0); // Convert seconds to milliseconds
      const response = await fetch("http://localhost:3001/manual_set_water_level", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          plant_type: plantType,
          growth_stage: growthStage,
          water_level: parseFloat(waterLevel),
          pump_duration_ms: parseInt(pumpDurationMs),
        }),
      });
      if (response.status === 401) {
        localStorage.removeItem("token");
        navigate("/login");
        return;
      }
      const data = await response.json();
      if (data.success) {
        alert(`Water level set to ${waterLevel}% (${waterAmount} mL) for ${plantType} at ${growthStage} stage. Pump activated for ${pumpTime} seconds.`);
        setIsPumpRunning(true);
        setPumpStartTime(Date.now()); // Store pump start time
      } else {
        alert("Failed to set water level: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error setting water level:", err);
      alert("Error setting water level: Server unreachable");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle stopping the pump
  const handleStopPump = async () => {
    if (!token) return navigate("/login");

    try {
      setIsSubmitting(true);
      const response = await fetch("http://localhost:3001/stop_pump", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
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
        setPumpStartTime(null); // Reset pump start time
      } else {
        alert("Failed to stop pump: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error stopping pump:", err);
      alert("Error stopping pump: Server unreachable");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Set standby mode when navigating back to home
  const handleBackToHome = async () => {
    if (!token) return navigate("/login");
    try {
      setIsSubmitting(true);
      const response = await fetch("http://localhost:3001/set_mode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ mode: "standby" }),
      });
      if (response.status === 401) {
        localStorage.removeItem("token");
        navigate("/login");
        return;
      }
      const data = await response.json();
      if (data.success) {
        navigate("/home");
      } else {
        console.error("Failed to set standby mode:", data.error);
        alert("Failed to set standby mode: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error setting standby mode:", err);
      alert("Error setting standby mode: Server unreachable");
      navigate("/home");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (loading) {
    return <div className="text-center p-6">Loading dataset...</div>;
  }

  return (
    <div className="relative min-h-screen">
      <img
        src="https://images.unsplash.com/photo-1508857650881-64475119d798?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        alt="background"
        className="absolute inset-0 w-full h-full object-cover z-0"
      />
    <div className="relative z-10 min-h-screen flex items-center justify-center bg-transparent p-6 bg-white/20 backdrop-blur-md">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md ">
        <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-black">Manual Water Level Set</h1>
        {espConnected === 'loading' && <p className="text-yellow-600 font-semibold mt-2">Checking connectivity...</p>}
        {espConnected === false && <p className="text-red-600 font-bold mt-2">ESP Devices are Not Connected!</p>}
        {espConnected === true && <p className="text-green-600 font-bold mt-2">Devices Connected and Ready</p>}
      </div>
        <div className="mb-4">
          <label className="block text-gray-700 text-lg mb-2">Plant Type</label>
          <select
            value={plantType}
            onChange={(e) => setPlantType(e.target.value)}
            className="w-full p-2 border rounded"
            disabled={isSubmitting}
          >
            <option value="Select">Select</option>
            <option value="Chilli">Chilli</option>
            <option value="Brinjal">Brinjal</option>
          </select>
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 text-lg mb-2">Growth Stage</label>
          <select
            value={growthStage}
            onChange={(e) => setGrowthStage(e.target.value)}
            className="w-full p-2 border rounded"
            disabled={isSubmitting}
          >
            <option value="Select">Select</option>
            <option value="Seedling">Seedling</option>
            <option value="Growing">Growing</option>
            <option value="Flowering">Flowering</option>
            <option value="Fruiting">Fruiting</option>
          </select>
        </div>
        <div className="mb-4">
          <div className="text-gray-700 text-sm bg-gray-200 p-2 rounded border">
            <p>Water Level: {waterLevelDisplay}</p>
            {waterAmount && <p>Water Amount: {waterAmount} mL</p>}
            {pumpTime && <p>Pumping Time: {pumpTime} seconds</p>}
          </div>
        </div>
        <div className="flex justify-center space-x-4">
          <button
            onClick={handleSetWaterLevel}
            className="bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-green-700 transition duration-200 disabled:bg-gray-400"
            disabled={espConnected !== true || isPumpRunning || isSubmitting || waterLevelDisplay === "No data available"}
          >
            {isSubmitting ? "Setting..." : "Set Water Level"}
          </button>
          <button
            onClick={handleStopPump}
            className="bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200 disabled:bg-gray-400"
            disabled={espConnected !== true || !isPumpRunning || isSubmitting}
          >
            {isSubmitting ? "Stopping..." : isPumpRunning ? "Stop Pump (Auto-stop after time)" : "Stop Pump"}
          </button>
          <button
            onClick={handleBackToHome}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200 disabled:bg-gray-400"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Navigating..." : "Back to Home"}
          </button>
        </div>
      </div>
    </div>
    </div>
  );
}