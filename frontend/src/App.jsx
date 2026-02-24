import React, { useState, useEffect } from 'react';
import ContextLayer from './components/ContextLayer';
import SystemLayer from './components/SystemLayer';
import IntelligenceLayer from './components/IntelligenceLayer';
import DecisionLayer from './components/DecisionLayer';
import './App.css';

function App() {
  // Simulate live data
  const [data, setData] = useState({
    context: {
      cpuUtil: 45,
      memoryUsage: 12.4,
      clockSpeed: 3.2,
      ambientTemp: 22,
      voltage: 1.15,
      current: 45.2
    },
    system: {
      currentTemp: 48,
      predictedTemp: 52
    },
    intelligence: {
      predictionHistory: []
    },
    decision: {
      status: 'STABLE',
      reason: 'All parameters within nominal operational ranges.',
      action: 'Maintain current cooling profile.'
    }
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/status');
        const json = await res.json();
        setData(json);
      } catch (e) {
        // Keep previous data on error
      }
    };

    fetchData();                          // Fire immediately on mount
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <div className="brand">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
          </svg>
          ANTIGRAVITY // THERMAL PREDICTOR
        </div>
        <div className="live-indicator">
          <div className="pulse"></div>
          SYSTEM ONLINE
        </div>
      </header>

      <main className="grid-layout">
        {/* Context Layer: Inputs */}
        <ContextLayer data={data.context} />

        {/* System Layer: Server Visualization */}
        <SystemLayer data={data.system} />

        {/* Intelligence & Decision Layers */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-xl)', height: '100%' }}>
          <IntelligenceLayer data={data.intelligence} />
          <DecisionLayer data={data.decision} />
        </div>
      </main>
    </div>
  );
}

export default App;
