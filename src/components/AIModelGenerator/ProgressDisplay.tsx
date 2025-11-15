'use client';

import React from 'react';

interface Step {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  details?: string;
}

interface ProgressDisplayProps {
  steps: Step[];
  currentStep: number;
  error?: string;
}

const STEP_ICONS = {
  'pending': '⏳',
  'in-progress': '⚙️',
  'completed': '✅',
  'error': '❌',
};

export default function ProgressDisplay({ steps, currentStep, error }: ProgressDisplayProps) {
  return (
    <div style={styles.container}>
      <style>{`
        .progress-display-title {
          font-size: 1.125rem;
          font-weight: 600;
          color: #ffffff;
          margin-bottom: 1rem;
        }

        .progress-display-steps {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .progress-display-step {
          padding: 1rem;
          border-radius: 0.5rem;
          border: 2px solid;
          transition: all 0.3s ease;
        }

        .progress-display-step.pending {
          border-color: #4b5563;
          background: #1f2937;
        }

        .progress-display-step.in-progress {
          border-color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
        }

        .progress-display-step.completed {
          border-color: #10b981;
          background: rgba(16, 185, 129, 0.1);
        }

        .progress-display-step.error {
          border-color: #ef4444;
          background: rgba(239, 68, 68, 0.1);
        }

        .progress-display-step-content {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
        }

        .progress-display-step-icon {
          font-size: 1.5rem;
          margin-top: 0.25rem;
          flex-shrink: 0;
        }

        .progress-display-step-text {
          flex: 1;
        }

        .progress-display-step-name {
          font-weight: 600;
          color: #ffffff;
          margin: 0;
        }

        .progress-display-step-details {
          font-size: 0.875rem;
          color: #d1d5db;
          margin-top: 0.25rem;
          margin-bottom: 0;
        }

        .progress-display-error {
          padding: 1rem;
          border-radius: 0.5rem;
          border: 2px solid #ef4444;
          background: rgba(239, 68, 68, 0.1);
          margin-top: 1rem;
        }

        .progress-display-error-title {
          font-weight: 600;
          color: #f87171;
          margin: 0 0 0.5rem 0;
        }

        .progress-display-error-message {
          font-size: 0.875rem;
          color: #fca5a5;
          margin: 0;
        }
      `}</style>

      <h3 className="progress-display-title">Training Progress</h3>

      <div className="progress-display-steps">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`progress-display-step ${step.status}`}
          >
            <div className="progress-display-step-content">
              <span className="progress-display-step-icon">
                {STEP_ICONS[step.status]}
              </span>
              <div className="progress-display-step-text">
                <h4 className="progress-display-step-name">{step.name}</h4>
                {step.details && (
                  <p className="progress-display-step-details">{step.details}</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="progress-display-error">
          <h4 className="progress-display-error-title">Error</h4>
          <p className="progress-display-error-message">{error}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
  },
};
