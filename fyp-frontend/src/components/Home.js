import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from '../firebase.config';
import { w3cwebsocket as W3CWebSocket } from "websocket";

export default function Home() {
  const navigate = useNavigate();
  const [waterLevel, setWaterLevel] = useState(null);
  const [status, setStatus] = useState('Monitoring...');

  useEffect(() => {
    const client = new W3CWebSocket('ws://localhost:5001');
    client.onopen = () => console.log('WebSocket Connected');
    client.onmessage = (message) => {
      const data = JSON.parse(message.data);
      setWaterLevel(data.waterLevel);
      setStatus(data.status || 'Monitoring...');
    };
    client.onerror = (error) => console.log('WebSocket Error:', error);

    return () => client.close();
  }, []);

  const handleSetWaterLevel = async () => {
    const token = await auth.currentUser.getIdToken();
    await fetch('http://localhost:5000/setWaterLevel', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ level: 50 }),
    });
  };

  const handleLogout = () => {
    auth.signOut().then(() => navigate("/login"));
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">FYP Water Level Monitor Dashboard</h1>
        <p className="text-gray-600 mt-2">Real-time monitoring and control</p>
      </div>

      {/* Dashboard Grid */}
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Water Level Card */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Water Level</h2>
          <p className="text-4xl font-bold text-green-600">
            {waterLevel !== null ? `${waterLevel}%` : 'Loading...'}
          </p>
          <p className="text-gray-500 mt-2">Last updated via WebSocket</p>
        </div>

        {/* Status Card */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">System Status</h2>
          <p className="text-2xl font-medium text-gray-800">{status}</p>
          <p className="text-gray-500 mt-2">Check sensor connection</p>
        </div>
      </div>

      {/* Control Section */}
      <div className="max-w-4xl mx-auto mt-8 text-center">
        <button
          onClick={handleSetWaterLevel}
          className="bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-green-700 transition duration-200"
        >
          Set Water Level to 50%
        </button>
      </div>

      {/* Logout Button */}
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