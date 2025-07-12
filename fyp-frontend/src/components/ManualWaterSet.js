import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";

export default function ManualWaterSet() {
  const navigate = useNavigate();
  const [plantType, setPlantType] = useState("Chilli");
  const [growthStage, setGrowthStage] = useState("Seedling");
  const [token] = useState(localStorage.getItem("token"));
  const [dataset, setDataset] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isPumpRunning, setIsPumpRunning] = useState(false);

  // Load and parse the CSV dataset
  useEffect(() => {
    fetch("/plant_water_dataset.csv")
      .then((response) => response.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: false,
          transformHeader: (header) => header.trim().replace(/^"|"$/g, ""),
          transform: (value) => value.trim().replace(/^"|"$/g, ""),
          complete: (results) => {
            const cleanedData = results.data.map((row) => ({
              growth_stage: row["growth_stage"],
              plant_type: row["plant_type"],
              water_level_percent: parseFloat(row["water_level_percent"]) || 0,
            }));
            setDataset(cleanedData);
            setLoading(false);
          },
          error: (err) => {
            console.error("Error parsing CSV:", err);
            setLoading(false);
          },
        });
      })
      .catch((err) => {
        console.error("Error fetching CSV:", err);
        setLoading(false);
      });
  }, []);

  // Calculate average water level
  const calculateAverageWaterLevel = () => {
    const filteredData = dataset.filter(
      (row) =>
        row.plant_type === plantType && row.growth_stage === growthStage
    );
    if (filteredData.length === 0) {
      console.warn("No data found for selected plant type and growth stage.");
      return 0;
    }
    const totalWaterLevel = filteredData.reduce(
      (sum, row) => sum + row.water_level_percent,
      0
    );
    return (totalWaterLevel / filteredData.length).toFixed(2);
  };

  // Handle setting water level
  const handleSetWaterLevel = async () => {
    if (!token) return navigate("/login");

    const waterLevel = calculateAverageWaterLevel();
    if (waterLevel <= 0) {
      alert("No valid water level data available for the selected options.");
      return;
    }

    try {
      const response = await fetch("http://192.168.56.1:3001/manual_set_water_level", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ plant_type: plantType, growth_stage: growthStage, water_level: waterLevel }),
      });
      const data = await response.json();
      if (data.success) {
        alert(
          `Water level set to ${waterLevel}% for ${plantType} at ${growthStage} stage. Pump activated.`
        );
        setIsPumpRunning(true);
      } else {
        alert("Failed to set water level: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Error setting water level:", err);
      alert("An error occurred while setting the water level.");
    }
  };

  // Handle stopping pump
  const handleStopPump = async () => {
    if (!token) return navigate("/login");

    try {
      const response = await fetch("http://192.168.56.1:3001/stop_pump", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
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

  if (loading) {
    return <div className="text-center p-6">Loading dataset...</div>;
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">Manual Water Level Set</h1>
        <p className="text-gray-600 mt-2">Set water level for testing</p>
      </div>
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
        <div className="mb-4">
          <label className="block text-gray-700 text-lg mb-2">Plant Type</label>
          <select
            value={plantType}
            onChange={(e) => setPlantType(e.target.value)}
            className="w-full p-2 border rounded"
          >
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
          >
            <option value="Seedling">Seedling</option>
            <option value="Growing">Growing</option>
            <option value="Flowering">Flowering</option>
            <option value="Fruiting">Fruiting</option>
          </select>
        </div>
        <div className="flex justify-center space-x-4">
          <button
            onClick={handleSetWaterLevel}
            className="bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-green-700 transition duration-200"
            disabled={isPumpRunning}
          >
            Set Water Level
          </button>
          <button
            onClick={handleStopPump}
            className="bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200"
            disabled={!isPumpRunning}
          >
            Stop Pump
          </button>
          <button
            onClick={() => navigate("/home")}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200"
          >
            Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}