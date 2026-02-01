import React, { useMemo } from 'react';
import { Server } from 'lucide-react';

const SystemLayer = ({ data }) => {
  const { currentTemp = 45, predictedTemp = 48 } = data || {};

  // Simple color interpolation logic or step-based
  const getTempColor = (temp) => {
    if (temp < 50) return 'var(--color-cool)';
    if (temp < 75) return 'var(--color-warm)';
    return 'var(--color-hot)';
  };

  const statusColor = getTempColor(currentTemp);

  return (
    <div className="panel" style={{ 
      position: 'relative', 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      border: '1px solid var(--border-subtle)',
      background: 'radial-gradient(circle at center, rgba(255,255,255,0.01) 0%, rgba(0,0,0,0) 70%)'
    }}>
      <div style={{ position: 'absolute', top: '1.5rem', left: '1.5rem' }}>
        <h2 style={{ 
          fontSize: '0.875rem', 
          fontWeight: 600, 
          color: 'var(--text-secondary)',
          letterSpacing: '0.05em' 
        }}>SYSTEM LAYER</h2>
      </div>

      {/* Visual Representation */}
      <div style={{ position: 'relative', width: '200px', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {/* Glow behind */}
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          background: statusColor,
          opacity: 0.1,
          filter: 'blur(40px)',
          borderRadius: '50%',
          transition: 'background-color 1s ease'
        }} />

        {/* Server Silhouette SVG */}
        <svg width="100%" height="100%" viewBox="0 0 200 300" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ transition: 'all 0.5s ease' }}>
           {/* Chassis Body */}
           <rect x="40" y="20" width="120" height="260" rx="4" stroke="var(--border-focus)" strokeWidth="2" fill="var(--bg-app)" />
           
           {/* Racks / Slots */}
           <rect x="55" y="40" width="90" height="10" rx="1" fill={statusColor} fillOpacity="0.2" />
           <rect x="55" y="60" width="90" height="10" rx="1" fill={statusColor} fillOpacity="0.2" />
           <rect x="55" y="80" width="90" height="10" rx="1" fill="var(--bg-card)" />
           
           <rect x="55" y="110" width="90" height="10" rx="1" fill="var(--bg-card)" />
           <rect x="55" y="130" width="90" height="10" rx="1" fill={statusColor} fillOpacity="0.3" />
           
           {/* Central Core (CPU Heat Source Visual) */}
           <circle cx="100" cy="190" r="30" fill="url(#heatGradient)" stroke={statusColor} strokeWidth="1" strokeOpacity="0.5" />
           
           <defs>
             <radialGradient id="heatGradient" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(100 190) rotate(90) scale(30)">
               <stop stopColor={statusColor} stopOpacity="0.4"/>
               <stop offset="1" stopColor={statusColor} stopOpacity="0"/>
             </radialGradient>
           </defs>
        </svg>

        {/* Floating Temp Badge (Minimal) */}
        <div style={{
          position: 'absolute',
          bottom: '20%',
          background: 'var(--bg-card)',
          border: `1px solid ${statusColor}`,
          padding: '4px 12px',
          borderRadius: '100px',
          display: 'flex',
          alignItems: 'baseline',
          gap: '4px'
        }}>
           <span className="text-value" style={{ fontSize: '1.5rem', fontWeight: 500 }}>{currentTemp}°C</span>
        </div>
      </div>

    </div>
  );
};

export default SystemLayer;
