import React, { useState } from "react";
  import { useNavigate } from "react-router-dom";
  import { auth, db } from "../firebase.config";
  import { createUserWithEmailAndPassword } from "firebase/auth";
  import { doc, setDoc, getDocs, query, collection, where } from "firebase/firestore";

  export default function Signup() {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);

    const validateUsername = async (username) => {
      if (username.length < 3) return "Username must be at least 3 characters long.";
      if (!/^[a-zA-Z0-9_]+$/.test(username)) return "Username can only contain letters, numbers, and underscores.";
      const q = query(collection(db, "users"), where("username", "==", username));
      const querySnapshot = await getDocs(q);
      if (!querySnapshot.empty) return "This username is already taken.";
      return null;
    };

    const handleSignup = async (e) => {
      e.preventDefault();
      if (!email || !username || !password) {
        alert("Please fill in all fields.");
        return;
      }

      setLoading(true);

      try {
        const usernameError = await validateUsername(username);
        if (usernameError) {
          alert(usernameError);
          setLoading(false);
          return;
        }

        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        await setDoc(doc(db, "users", user.uid), {
          username,
          email,
          createdAt: new Date().toISOString(),
        });

        console.log("User signed up:", user.uid);
        navigate("/home");
      } catch (error) {
        let errorMessage = "An error occurred during signup.";
        switch (error.code) {
          case "auth/email-already-in-use":
            errorMessage = "This email is already in use.";
            break;
          case "auth/invalid-email":
            errorMessage = "Please enter a valid email address.";
            break;
          case "auth/weak-password":
            errorMessage = "Password should be at least 6 characters.";
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
            <h1 className="text-4xl text-white font-bold mb-2 shadow-md">Create Account</h1>
            <p className="text-base text-gray-300 font-normal mb-7">Join us to start protecting your crops</p>
            <form onSubmit={handleSignup} className="mb-5">
              <input
                type="email"
                className="w-full bg-white bg-opacity-10 text-white rounded-2xl p-3 mb-4 text-base font-normal shadow-md"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="off"
                disabled={loading}
              />
              <input
                type="text"
                className="w-full bg-white bg-opacity-10 text-white rounded-2xl p-3 mb-4 text-base font-normal shadow-md"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
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
                {loading ? <span className="loading loading-spinner"></span> : "Sign Up"}
              </button>
            </form>
            <div className="flex justify-center mt-5">
              <p className="text-gray-300 text-base font-normal">Already have an account? </p>
              <button onClick={() => navigate("/login")} className="text-green-400 text-base font-semibold underline ml-1">
                Log in
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }