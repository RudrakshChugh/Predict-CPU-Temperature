import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';

const DecisionLayer = ({ data }) => {
  const { status = 'STABLE', reason, action } = data || {
    status: 'WATCH',
    reason: 'Projected temperature nearing 70°C threshold within 90 seconds.',
    action: 'Throttle non-critical background processes.'
  };

  const statusConfig = {
    STABLE: { color: 'var(--status-stable)', icon: CheckCircle },
    WATCH: { color: 'var(--status-watch)', icon: AlertTriangle },
    CRITICAL: { color: 'var(--status-critical)', icon: AlertCircle },
  };

  const currentConfig = statusConfig[status] || statusConfig.STABLE;
  const Icon = currentConfig.icon;

  return (
    <div className="panel" style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '50%',
      borderTop: 'none', /* If stacked, visual separation styling */
      background: 'linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-card) 100%)'
    }}>
      <div style={{ marginBottom: 'auto' }}>
        <h2 style={{ 
          fontSize: '0.875rem', 
          fontWeight: 600, 
          color: 'var(--text-secondary)',
          letterSpacing: '0.05em',
          marginBottom: '1rem'
        }}>DECISION LAYER</h2>

        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '12px', 
          marginBottom: '1.5rem',
          padding: '1rem',
          border: `1px solid ${currentConfig.color}`,
          background: `color-mix(in srgb, ${currentConfig.color} 5%, transparent)`,
          borderRadius: '4px'
        }}>
          <Icon color={currentConfig.color} size={24} />
          <span style={{ 
            color: currentConfig.color, 
            fontWeight: 700, 
            letterSpacing: '0.1em',
            fontSize: '1.25rem' 
          }}>
            {status}
          </span>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <div className="text-label" style={{ marginBottom: '0.5rem' }}>Analysis</div>
          <p style={{ color: 'var(--text-primary)', lineHeight: '1.6' }}>{reason}</p>
        </div>

        <div>
          <div className="text-label" style={{ marginBottom: '0.5rem' }}>Recommended Action</div>
          <p style={{ color: 'var(--text-primary)', lineHeight: '1.6', fontWeight: 500 }}>{action}</p>
        </div>
      </div>
    </div>
  );
};

export default DecisionLayer;
