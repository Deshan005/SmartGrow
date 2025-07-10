import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./components/Home";
import UserDetails from "./components/UserDetails";
import Signup from "./components/Signup";
import Login from "./components/Login";
import ManualWaterSet from "./components/ManualWaterSet"

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/home" element={<Home />} />
        <Route path="/user-details" element={<UserDetails />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/manual-water-set" element={<ManualWaterSet />} />
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Login />} />
      </Routes>
    </Router>
  );
}

export default App;