import React, { useState } from "react";
import logo from './logo.svg';
import './App.css';
import CameraCapture from './components/CameraCapture';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <CameraCapture />
      </header>
    </div>
  );
}

export default App;
