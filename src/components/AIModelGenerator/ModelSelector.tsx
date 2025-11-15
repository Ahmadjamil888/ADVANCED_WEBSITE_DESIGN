'use client';

import React from 'react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

const AVAILABLE_MODELS = [
  {
    id: 'mixtral-8x7b-32768',
    name: 'Mixtral 8x7B',
    description: 'Fast and efficient, great for code generation',
    icon: '⚡',
  },
  {
    id: 'llama2-70b-4096',
    name: 'Llama 2 70B',
    description: 'Powerful model for complex tasks',
    icon: '🦙',
  },
  {
    id: 'gemma-7b-it',
    name: 'Gemma 7B',
    description: 'Lightweight and fast',
    icon: '💎',
  },
];

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div style={styles.container}>
      <style>{`
        .model-selector-label {
          display: block;
          font-size: 0.875rem;
          font-weight: 600;
          color: #d1d5db;
          margin-bottom: 1rem;
        }

        .model-selector-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
        }

        .model-selector-button {
          padding: 1rem;
          border-radius: 0.5rem;
          border: 2px solid;
          transition: all 0.3s ease;
          text-align: left;
          background: #1f2937;
          border-color: #4b5563;
          cursor: pointer;
          font-family: inherit;
        }

        .model-selector-button:hover {
          border-color: #6b7280;
          background: #374151;
        }

        .model-selector-button.selected {
          border-color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
        }

        .model-selector-button-content {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
        }

        .model-selector-icon {
          font-size: 1.5rem;
          flex-shrink: 0;
        }

        .model-selector-text {
          flex: 1;
        }

        .model-selector-name {
          font-weight: 600;
          color: #ffffff;
          margin: 0;
          margin-bottom: 0.25rem;
        }

        .model-selector-description {
          font-size: 0.875rem;
          color: #9ca3af;
          margin: 0;
        }

        @media (max-width: 768px) {
          .model-selector-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <label className="model-selector-label">Select AI Model</label>
      <div className="model-selector-grid">
        {AVAILABLE_MODELS.map((model) => (
          <button
            key={model.id}
            onClick={() => onModelChange(model.id)}
            className={`model-selector-button ${selectedModel === model.id ? 'selected' : ''}`}
          >
            <div className="model-selector-button-content">
              <span className="model-selector-icon">{model.icon}</span>
              <div className="model-selector-text">
                <h3 className="model-selector-name">{model.name}</h3>
                <p className="model-selector-description">{model.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
  },
};
