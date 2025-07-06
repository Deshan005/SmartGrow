import React, { useState } from "react";
  import { useNavigate } from "react-router-dom";
  import { auth, db } from "../firebase.config";
  import { signInWithEmailAndPassword } from "firebase/auth";
  import { query, collection, where, getDocs } from "firebase/firestore";
 

  export default function Login() {
    const navigate = useNavigate();
    const [identifier, setIdentifier] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);

    const handleLogin = async (e) => {
      e.preventDefault();
      if (!identifier || !password) {
        alert("Please fill in all fields.");
        return;
      }

      setLoading(true);

      try {
        let email = identifier;

        if (!identifier.includes("@")) {
          const q = query(collection(db, "users"), where("username", "==", identifier));
          const querySnapshot = await getDocs(q);

          if (querySnapshot.empty) {
            throw new Error("auth/user-not-found");
          }

          const userDoc = querySnapshot.docs[0];
          email = userDoc.data().email;
        }

        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        console.log("User logged in:", user.uid);
        navigate("/home");
      } catch (error) {
        let errorMessage = "An error occurred during login.";
        switch (error.code || error.message) {
          case "auth/user-not-found":
            errorMessage = "No user found with this email or username.";
            break;
          case "auth/wrong-password":
            errorMessage = "Incorrect password.";
            break;
          case "auth/invalid-email":
            errorMessage = "Please enter a valid email address.";
            break;
          default:
            errorMessage = error.message;
        }
        alert(errorMessage);
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
            <h1 className="text-4xl text-white font-bold mb-2 shadow-md">Welcome Back</h1>
            <p className="text-base text-gray-300 font-normal mb-7">Please sign in to continue</p>
            <form onSubmit={handleLogin} className="mb-5">
              <input
                type="text"
                className="w-full bg-white bg-opacity-10 text-white rounded-2xl p-3 mb-4 text-base font-normal shadow-md"
                placeholder="Email or Username"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                autoComplete="off"
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
                {loading ? <span className="loading loading-spinner"></span> : "Login"}
              </button>
            </form>
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