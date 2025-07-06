import React from 'react';
  import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
  import Login from './components/Login';
  import Signup from './components/Signup';
  import Home from './components/Home';

  function App() {
    return (
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/home" element={<Home />} />
          <Route path="/" element={<Login />} />
        </Routes>
      </Router>
    );
  }

  export default App;