import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function ManualWaterSet() {
  const navigate = useNavigate();
  const [plantType, setPlantType] = useState('Chilli');
  const [growthStage, setGrowthStage] = useState('Seedling');
  const [token] = useState(localStorage.getItem('token'));

  const handleSetWaterLevel = async () => { 
    if (!token) return navigate('/login');
    const response = await fetch('http://localhost:5000/manual_set_water_level', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ plant_type: plantType, growth_stage: growthStage }),
    });
    const data = await response.json();
    if (data.success) {
      alert(`Water level set to ${data.water_level}%`);
    }
  };

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
        <button
          onClick={handleSetWaterLevel}
          className="bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-green-700 transition duration-200"
        >
          Set Water Level
        </button>
        <button
          onClick={() => navigate('/home')}
          className="ml-4 bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200"
        >
          Back to Home
        </button>
      </div>
    </div>
  );
}