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

      localStorage.setItem('token', data.token);
      navigate("/home");
    } catch (error) {
      setErrorMessage("An error occurred during login.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen">
      <img
        src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=1920"
        alt="background"
        className="absolute inset-0 w-full h-full object-cover z-0"
      />
      <div className="relative z-10 min-h-screen flex items-center justify-center p-6 bg-white/0 backdrop-blur-md">
        <div className="w-full max-w-md bg-white/50 rounded-xl shadow-2xl p-8">
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800">Login</h1>
            <p className="text-gray-600 mt-2">Welcome back to protect your crops</p>
          </div>
          <form onSubmit={handleLogin}>
            <input
              type="text"
              className="w-full mb-4 px-4 py-3 rounded-lg text-gray-800 bg-white border shadow-inner focus:outline-none focus:ring-2 focus:ring-green-400"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
            />
            <input
              type="password"
              className="w-full mb-4 px-4 py-3 rounded-lg text-gray-800 bg-white border shadow-inner focus:outline-none focus:ring-2 focus:ring-green-400"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 rounded-lg font-semibold text-white shadow-lg transition duration-200 ${loading ? "bg-green-700 opacity-70" : "bg-green-600 hover:bg-green-700"}`}
            >
              {loading ? "Logging in..." : "Log In"}
            </button>
          </form>
          {errorMessage && (
            <p className="text-red-500 text-center text-sm mt-4">{errorMessage}</p>
          )}
          <div className="flex justify-center mt-6 text-sm text-gray-700">
            <p>Don't have an account?</p>
            <button
              onClick={() => navigate("/signup")}
              className="ml-1 text-green-600 font-semibold hover:underline"
            >
              Sign up
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
