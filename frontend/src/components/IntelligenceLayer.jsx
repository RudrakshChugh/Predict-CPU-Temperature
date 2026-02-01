import React from 'react';
import { AreaChart, Area, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', padding: '8px 12px', fontSize: '12px' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Time: {payload[0].payload.time}s</p>
        <p style={{ color: 'var(--text-primary)' }}>Temp: {payload[0].value}°C</p>
      </div>
    );
  }
  return null;
};

const IntelligenceLayer = ({ data }) => {
  // Mock prediction series if not provided
  // Should accept an array of { time, temp, isPredicted }
  const { predictionHistory = [] } = data || {};

  // Find peak in prediction
  const peak = predictionHistory.length > 0 
    ? Math.max(...predictionHistory.map(d => d.temp)) 
    : 0;
    
  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', height: '50%', minHeight: '300px' }}>
      <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
           <h2 style={{ 
            fontSize: '0.875rem', 
            fontWeight: 600, 
            color: 'var(--text-secondary)',
            letterSpacing: '0.05em' 
          }}>INTELLIGENCE LAYER</h2>
          <p className="text-label" style={{ marginTop: '0.25rem', textTransform: 'none' }}>Short-horizon Forecast (5 min)</p>
        </div>
        <div style={{ textAlign: 'right' }}>
           <div className="text-label">Predicted Peak</div>
           <div className="text-mono" style={{ color: 'var(--text-primary)' }}>{peak}°C</div>
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={predictionHistory}>
            <defs>
              <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--text-mono)" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="var(--text-mono)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="time" hide />
            <YAxis domain={['dataMin - 5', 'dataMax + 5']} hide />
            <ReferenceLine y={80} stroke="var(--color-hot)" strokeDasharray="3 3" />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border-focus)', strokeWidth: 1 }} />
            <Area 
              type="monotone" 
              dataKey="temp" 
              stroke="var(--text-mono)" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorTemp)" 
              animationDuration={1000}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        "Projected thermal trajectory under current workload."
      </div>
    </div>
  );
};

export default IntelligenceLayer;
