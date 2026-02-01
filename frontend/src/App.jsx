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
    // Simulation Tick
    const interval = setInterval(() => {
      setData(prev => {
        // Random walk simulation
        const cpuUtil = Math.min(100, Math.max(0, prev.context.cpuUtil + (Math.random() - 0.5) * 5));
        const tempChange = (cpuUtil - 50) * 0.05;
        const newTemp = Math.min(95, Math.max(20, prev.system.currentTemp + tempChange));
        
        // Generate history point
        const newPoint = { time: new Date().getSeconds(), temp: newTemp };
        const history = [...(prev.intelligence.predictionHistory || []), newPoint].slice(-20);

        // Determine State
        let status = 'STABLE';
        let reason = 'System operating efficiently.';
        let action = 'No action required.';

        if (newTemp > 75) {
          status = 'CRITICAL';
          reason = 'Thermal runaway detected due to sustained high utilization.';
          action = 'IMMEDIATE: Throttle CPU voltage and increase fan speed to 100%.';
        } else if (newTemp > 60) {
          status = 'WATCH';
          reason = 'Temperature rising above optimal baseline.';
          action = 'Increase fan speed by 20% preemptively.';
        }

        return {
          context: {
            ...prev.context,
            cpuUtil: Math.floor(cpuUtil),
            memoryUsage: 12 + Math.random(),
            clockSpeed: 3.2 + (Math.random() - 0.5) * 0.1,
            voltage: 1.1 + (Math.random() - 0.5) * 0.05,
            current: 40 + (cpuUtil * 0.5)
          },
          system: {
            currentTemp: Math.floor(newTemp),
            predictedTemp: Math.floor(newTemp + (Math.random() * 5))
          },
          intelligence: {
            predictionHistory: history
          },
          decision: {
            status,
            reason,
            action
          }
        };
      });
    }, 1000);

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
