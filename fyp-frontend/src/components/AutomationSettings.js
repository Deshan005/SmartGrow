import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function AutomationSettings() {
  const navigate = useNavigate();
  const [currentCrops, setCurrentCrops] = useState([]);
  const [plantType, setPlantType] = useState('Chilli');
  const [seedDate, setSeedDate] = useState(new Date().toISOString().split('T')[0]);
  const [plantStatus, setPlantStatus] = useState('');
  const [savingPlant, setSavingPlant] = useState(false);
  const token = localStorage.getItem('token');

  useEffect(() => {
    const fetchPlantData = async () => {
      if (!token) return navigate('/login');
      try {
        const plantResponse = await fetch('http://localhost:3001/user_plants', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (plantResponse.ok) {
           const plantData = await plantResponse.json();
           if (plantData.plants && plantData.plants.length > 0) {
              const formattedCrops = plantData.plants.map(p => ({
                 type: p.plant_type,
                 seedDate: new Date(p.seed_date).toLocaleDateString()
              }));
              setCurrentCrops(formattedCrops);
           }
        } else {
           if(plantResponse.status === 401) navigate('/login');
        }
      } catch (error) {
        console.error(error);
      }
    };
    fetchPlantData();
  }, [navigate, token]);

  const handleStartCrop = async () => {
    if (!plantType || !seedDate) {
      setPlantStatus('Please select a plant and date.');
      return;
    }
    setSavingPlant(true);
    setPlantStatus('');
    try {
      const response = await fetch('http://localhost:3001/user_plant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ plant_type: plantType, seed_date: seedDate })
      });
      const data = await response.json();
      if (response.ok) {
        setPlantStatus('Crop tracking updated successfully!');
        
        // update the local array
        setCurrentCrops(prev => {
          const formattedDate = new Date(seedDate).toLocaleDateString();
          const exists = prev.find(p => p.type === plantType);
          if (exists) {
            return prev.map(p => p.type === plantType ? { ...p, seedDate: formattedDate } : p);
          } else {
            return [...prev, { type: plantType, seedDate: formattedDate }];
          }
        });
      } else {
        setPlantStatus(data.error || 'Failed to update crop.');
      }
    } catch (err) {
      setPlantStatus('Server error. Could not update crop.');
    } finally {
      setSavingPlant(false);
    }
  };

  return (
    <div className="relative min-h-screen">
      <img
        src="https://images.unsplash.com/photo-1508857650881-64475119d798?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        alt="background"
        className="absolute inset-0 w-full h-full object-cover z-0"
      />
      <div className="relative z-10 min-h-screen flex items-center justify-center bg-transparent p-6 bg-white/20 backdrop-blur-md">
        <div className="relative z-10 w-full max-w-2xl px-4">
          <div className="bg-white/90 p-6 rounded-lg shadow-md text-center max-w-md mx-auto w-full mb-6">
             <h1 className="text-3xl font-bold text-gray-800 mb-4">Automation Settings</h1>
             
             <div className="pt-4 text-left">
                <h2 className="text-xl font-semibold text-green-800 mb-3 text-center">Current Automated Crops</h2>
                {currentCrops.length > 0 ? (
                  <div className="grid grid-cols-1 gap-3 mb-4">
                     {currentCrops.map((crop, idx) => (
                       <div key={idx} className="bg-green-100 p-3 rounded-lg text-center border border-green-300 shadow-sm">
                          <p className="text-lg font-bold text-green-900">{crop.type}</p>
                          <p className="text-sm text-green-800">Planted on: {crop.seedDate}</p>
                       </div>
                     ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center italic mb-4">No crops are currently being automatically tracked.</p>
                )}

                <h3 className="font-semibold text-gray-700 text-center mb-2">Start / Update Tracking</h3>
                <div className="flex flex-col space-y-3">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Crop Type</label>
                    <select value={plantType} onChange={(e) => setPlantType(e.target.value)} className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-400">
                      <option value="Chilli">Chilli</option>
                      <option value="Brinjal">Brinjal</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Seed Date</label>
                    <input type="date" value={seedDate} onChange={(e) => setSeedDate(e.target.value)} className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-green-400" />
                  </div>
                  <button 
                     onClick={handleStartCrop}
                     disabled={savingPlant}
                     className="bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg font-semibold transition shadow-md"
                  >
                    {savingPlant ? 'Saving...' : 'Set Automation Tracking'}
                  </button>
                  {plantStatus && <p className="text-center text-sm font-medium text-green-700 mt-2">{plantStatus}</p>}
                </div>
             </div>

             <div className="flex justify-center mt-6">
               <button
                 onClick={() => navigate('/home')}
                 className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200 w-full"
               >
                 Dashboard
               </button>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}
