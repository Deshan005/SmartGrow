import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setErrorMessage("Please fill in all fields.");
      return;
    }

    setLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch('http://localhost:3001/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();

      if (!response.ok) {
        setErrorMessage(data.error || 'Login failed');
        return;
      }

      localStorage.setItem('token', data.token); // Store JWT token
      navigate("/home");
    } catch (error) {
      setErrorMessage("An error occurred during login.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black relative">
      <img
        src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=1920"
        alt="background"
        className="absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black opacity-70"></div>
      <div className="flex items-center justify-center min-h-screen p-6 pt-36 pb-10 relative z-10">
        <div className="text-center">
          <h1 className="text-4xl text-white font-bold mb-2 shadow-md">Login</h1>
          <p className="text-base text-gray-300 font-normal mb-7">Welcome back to protect your crops</p>
          <form onSubmit={handleLogin} className="mb-5">
            <input
              type="text"
              className="w-full bg-white bg-opacity-10 text-white rounded-2xl p-3 mb-4 text-base font-normal shadow-md"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
            />
            <input
              type="password"
              className="w-full bg-white bg-opacity-10 text-white rounded-2xl p-3 mb-4 text-base font-normal shadow-md"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              className={`w-full bg-green-600 p-4 rounded-2xl text-white text-lg font-semibold shadow-lg ${loading ? 'bg-green-700 opacity-70' : ''}`}
              disabled={loading}
            >
              {loading ? <span className="loading loading-spinner"></span> : "Log In"}
            </button>
          </form>
          {errorMessage && <p className="text-red-400 text-base font-semibold mt-2">{errorMessage}</p>}
          <div className="flex justify-center mt-5">
            <p className="text-gray-300 text-base font-normal">Don't have an account? </p>
            <button onClick={() => navigate("/signup")} className="text-green-400 text-base font-semibold underline ml-1">
              Sign up
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}