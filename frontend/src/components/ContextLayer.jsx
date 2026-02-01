import React from 'react';
import { Cpu, Zap, Thermometer, Box, Activity } from 'lucide-react';
import '../index.css';

const MetricRow = ({ icon: Icon, label, value, unit }) => (
  <div style={{ 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'space-between', 
    marginBottom: '1.5rem',
    opacity: 0.8
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
      <Icon size={16} color="var(--text-muted)" />
      <span className="text-label">{label}</span>
    </div>
    <div>
      <span className="text-value">{value}</span>
      <span className="text-mono" style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginLeft: '4px' }}>{unit}</span>
    </div>
  </div>
);

const ContextLayer = ({ data }) => {
  // Destructure with default values
  const {
    cpuUtil = 0,
    memoryUsage = 0,
    clockSpeed = 0.0,
    ambientTemp = 0,
    voltage = 0.0,
    current = 0.0,
  } = data || {};

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ marginBottom: '2rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem' }}>
        <h2 style={{ 
          fontSize: '0.875rem', 
          fontWeight: 600, 
          color: 'var(--text-secondary)',
          letterSpacing: '0.05em' 
        }}>CONTEXT LAYER</h2>
        <p className="text-label" style={{ marginTop: '0.5rem', textTransform: 'none' }}>Real-time Input Parameters</p>
      </div>

      <div style={{ flex: 1 }}>
        <MetricRow icon={Cpu} label="CPU Load" value={cpuUtil} unit="%" />
        <MetricRow icon={Box} label="Memory" value={memoryUsage} unit="GB" />
        <MetricRow icon={Activity} label="Clock Speed" value={clockSpeed.toFixed(2)} unit="GHz" />
        <MetricRow icon={Thermometer} label="Ambient Temp" value={ambientTemp} unit="°C" />
        <MetricRow icon={Zap} label="Voltage" value={voltage.toFixed(2)} unit="V" />
        <MetricRow icon={Zap} label="Current" value={current.toFixed(1)} unit="A" />
      </div>

      <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
        <div className="text-label">System Uptime</div>
        <div className="text-mono" style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>14d 03h 22m</div>
      </div>
    </div>
  );
};

export default ContextLayer;
