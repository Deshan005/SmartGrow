import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCpir0yOjBfERR9ME7FeDK3Pf-iS4jtdtc",
  authDomain: "smartgrow-7daab.firebaseapp.com",
  projectId: "smartgrow-7daab",
  storageBucket: "smartgrow-7daab.firebasestorage.app",
  messagingSenderId: "546070208500",
  appId: "1:546070208500:web:01c7af006bcb4aa7d4e3a1",
  measurementId: "G-35JBX7CPN9"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
export { app, auth, db };