'use client';

import React from 'react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

const AVAILABLE_MODELS = [
  {
    id: 'llama3-70b-8192',
    name: 'Llama 3 70B',
    description: 'Powerful model for code generation',
  },
  {
    id: 'llama3-8b-8192',
    name: 'Llama 3 8B',
    description: 'Fast and lightweight',
  },
  {
    id: 'gemma2-9b-it',
    name: 'Gemma 2 9B',
    description: 'Balanced performance',
  },
  {
    id: 'llama-3.1-8b-instant',
    name: 'Llama 3.1 8B Instant',
    description: 'Ultra-fast responses',
  },
];

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div style={styles.container}>
      <style>{`
        .model-selector-wrapper {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .model-selector-label {
          font-size: 0.875rem;
          font-weight: 600;
          color: #ffffff;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .model-selector-toggle-group {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }

        .model-selector-toggle {
          padding: 0.75rem 1.25rem;
          background: #1a1a1a;
          border: 1px solid #333333;
          color: #999999;
          cursor: pointer;
          font-size: 0.9rem;
          transition: all 0.2s ease;
          font-family: inherit;
          font-weight: 500;
          white-space: nowrap;
        }

        .model-selector-toggle:hover {
          background: #222222;
          border-color: #444444;
          color: #ffffff;
        }

        .model-selector-toggle.active {
          background: #ffffff;
          border-color: #ffffff;
          color: #000000;
        }

        .model-selector-info {
          padding: 1rem;
          background: #0a0a0a;
          border: 1px solid #222222;
          border-radius: 1px;
        }

        .model-selector-info-name {
          font-weight: 600;
          color: #ffffff;
          margin: 0 0 0.5rem 0;
          font-size: 0.95rem;
        }

        .model-selector-info-desc {
          color: #999999;
          margin: 0;
          font-size: 0.85rem;
        }

        @media (max-width: 768px) {
          .model-selector-toggle-group {
            flex-direction: column;
          }

          .model-selector-toggle {
            width: 100%;
          }
        }
      `}</style>

      <div className="model-selector-wrapper">
        <label className="model-selector-label">Select Model</label>
        <div className="model-selector-toggle-group">
          {AVAILABLE_MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => onModelChange(model.id)}
              className={`model-selector-toggle ${selectedModel === model.id ? 'active' : ''}`}
            >
              {model.name}
            </button>
          ))}
        </div>
        
        {/* Show info about selected model */}
        {AVAILABLE_MODELS.find(m => m.id === selectedModel) && (
          <div className="model-selector-info">
            <p className="model-selector-info-name">
              {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}
            </p>
            <p className="model-selector-info-desc">
              {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.description}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
  },
};
