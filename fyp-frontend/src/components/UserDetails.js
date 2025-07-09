import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function UserDetails() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const token = localStorage.getItem('token');

  useEffect(() => {
    const fetchUserDetails = async () => {
      if (!token) return navigate('/login');
      try {
        const response = await fetch('http://localhost:5000/user-details', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
          setUser({
            username: data.username || 'No Username',
            email: data.email || 'No Email',
            joinDate: data.created_at ? new Date(data.created_at).toLocaleDateString() : 'N/A',
          });
        } else {
          navigate('/login');
        }
      } catch (error) {
        navigate('/login');
      }
    };
    fetchUserDetails();
  }, [navigate, token]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate("/login");
  };

  if (!user) return <div className="text-center mt-10 text-gray-600">Loading...</div>;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto text-center">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">User Profile</h1>
          <p className="text-xl text-gray-700 mb-2">Username: {user.username}</p>
          <p className="text-lg text-gray-600 mb-2">Email: {user.email}</p>
          <p className="text-lg text-gray-600 mb-4">Joined: {user.joinDate}</p>
          <div className="flex justify-center space-x-4 mt-4">
            <button
              onClick={() => navigate('/home')}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-blue-700 transition duration-200"
            >
              Back to Home
            </button>
            <button
              onClick={handleLogout}
              className="bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-semibold shadow-md hover:bg-red-700 transition duration-200"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}