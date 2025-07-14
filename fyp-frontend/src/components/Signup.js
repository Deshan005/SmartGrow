import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Signup() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleSignup = async (e) => {
    e.preventDefault();
    if (!email || !username || !password) {
      setErrorMessage("Please fill in all fields.");
      return;
    }

    setLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch("http://localhost:3001/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, username, password }),
      });
      const data = await response.json();

      if (!response.ok) {
        setErrorMessage(data.error || "Signup failed");
        return;
      }

      // Store JWT token if returned (optional, depending on backend)
      if (data.token) {
        localStorage.setItem("token", data.token);
      }
      setTimeout(() => {
        navigate("/login");
      }, 1000);
    } catch (error) {
      setErrorMessage("An error occurred during signup.");
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
            <h1 className="text-3xl font-bold text-gray-800">Sign Up</h1>
            <p className="text-gray-600 mt-2">Join us to protect your crops</p>
          </div>
          <form onSubmit={handleSignup}>
            <input
              type="email"
              className="w-full mb-4 px-4 py-3 rounded-lg text-gray-800 bg-white border shadow-inner focus:outline-none focus:ring-2 focus:ring-green-400"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
            />
            <input
              type="text"
              className="w-full mb-4 px-4 py-3 rounded-lg text-gray-800 bg-white border shadow-inner focus:outline-none focus:ring-2 focus:ring-green-400"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="off"
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
              {loading ? "Signing up..." : "Sign Up"}
            </button>
          </form>
          {errorMessage && (
            <p className="text-red-500 text-center text-sm mt-4">{errorMessage}</p>
          )}
          <div className="flex justify-center mt-6 text-sm text-gray-700">
            <p>Already have an account?</p>
            <button
              onClick={() => navigate("/login")}
              className="ml-1 text-green-600 font-semibold hover:underline"
              disabled={loading}
            >
              Log In
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}