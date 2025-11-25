'use client';

import React, { useState } from 'react';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  isLoading: boolean;
}

const EXAMPLE_PROMPTS = [
  'Create a sentiment analysis model that can classify text as positive, negative, or neutral',
  'Build a neural network for image classification using CIFAR-10 dataset',
  'Generate a time series forecasting model for stock price prediction',
  'Create a text generation model using transformer architecture',
];

export default function PromptInput({ onSubmit, isLoading }: PromptInputProps) {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt);
      setPrompt('');
    }
  };

  const handleExampleClick = (example: string) => {
    setPrompt(example);
  };

  return (
    <div style={styles.container}>
      <style>{`
        .prompt-input-form {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .prompt-input-group {
          display: flex;
          flex-direction: column;
        }

        .prompt-input-label {
          display: block;
          font-size: 0.875rem;
          font-weight: 600;
          color: #d1d5db;
          margin-bottom: 0.5rem;
        }

        .prompt-input-textarea {
          width: 100%;
          height: 8rem;
          padding: 1rem;
          background: #1f2937;
          border: 1px solid #4b5563;
          border-radius: 0.5rem;
          color: #ffffff;
          font-family: inherit;
          font-size: 1rem;
          resize: none;
          transition: all 0.3s ease;
        }

        .prompt-input-textarea::placeholder {
          color: #6b7280;
        }

        .prompt-input-textarea:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .prompt-input-textarea:disabled {
          background: #111827;
          color: #6b7280;
          cursor: not-allowed;
        }

        .prompt-input-button {
          width: 100%;
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          font-weight: 600;
          transition: all 0.3s ease;
          border: none;
          cursor: pointer;
          font-size: 1rem;
        }

        .prompt-input-button:disabled {
          background: #374151;
          color: #9ca3af;
          cursor: not-allowed;
        }

        .prompt-input-button:not(:disabled) {
          background: #2563eb;
          color: white;
        }

        .prompt-input-button:not(:disabled):hover {
          background: #1d4ed8;
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        .prompt-input-button-content {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .prompt-input-spinner {
          display: inline-block;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .prompt-input-examples {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .prompt-input-examples-label {
          font-size: 0.875rem;
          color: #9ca3af;
        }

        .prompt-input-examples-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 0.5rem;
        }

        .prompt-input-example-button {
          padding: 0.75rem;
          text-align: left;
          font-size: 0.875rem;
          background: #1f2937;
          border: 1px solid #4b5563;
          border-radius: 0.375rem;
          color: #d1d5db;
          cursor: pointer;
          transition: all 0.3s ease;
          font-family: inherit;
        }

        .prompt-input-example-button:hover {
          background: #374151;
          border-color: #6b7280;
          color: #ffffff;
        }

        @media (max-width: 768px) {
          .prompt-input-examples-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <form onSubmit={handleSubmit} className="prompt-input-form">
        <div className="prompt-input-group">
          <label className="prompt-input-label">Describe Your AI Model</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Create a sentiment analysis model that classifies text as positive, negative, or neutral..."
            className="prompt-input-textarea"
            disabled={isLoading}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className="prompt-input-button"
        >
          <div className="prompt-input-button-content">
            {isLoading && <span className="prompt-input-spinner">⚙️</span>}
            <span>{isLoading ? 'Generating & Training Model...' : 'Generate & Train Model'}</span>
          </div>
        </button>
      </form>

      {!isLoading && (
        <div className="prompt-input-examples">
          <p className="prompt-input-examples-label">Try an example:</p>
          <div className="prompt-input-examples-grid">
            {EXAMPLE_PROMPTS.map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                className="prompt-input-example-button"
              >
                {example}
              </button>
            ))}
          </div>
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
