'use client';

import React, { useState } from 'react';

interface DeploymentResultProps {
  deploymentUrl: string;
  sandboxId: string;
  modelType: string;
  endpoints: {
    health: string;
    predict: string;
    info: string;
  };
}

export default function DeploymentResult({
  deploymentUrl,
  sandboxId,
  modelType,
  endpoints,
}: DeploymentResultProps) {
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  const copyToClipboard = (text: string, endpoint: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(endpoint);
    setTimeout(() => setCopiedEndpoint(null), 2000);
  };

  return (
    <div style={styles.container}>
      <style>{`
        .deployment-result-wrapper {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .deployment-result-success {
          padding: 1.5rem;
          border-radius: 0.5rem;
          background: rgba(16, 185, 129, 0.1);
          border: 2px solid #10b981;
        }

        .deployment-result-success-title {
          font-size: 1.5rem;
          font-weight: 700;
          color: #4ade80;
          margin: 0 0 0.5rem 0;
        }

        .deployment-result-success-text {
          color: #d1d5db;
          margin: 0;
        }

        .deployment-result-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
        }

        .deployment-result-card {
          padding: 1rem;
          border-radius: 0.5rem;
          background: #1f2937;
          border: 1px solid #4b5563;
        }

        .deployment-result-card-label {
          display: block;
          font-size: 0.875rem;
          font-weight: 600;
          color: #9ca3af;
          margin-bottom: 0.5rem;
        }

        .deployment-result-card-content {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .deployment-result-code {
          flex: 1;
          padding: 0.5rem;
          background: #111827;
          border-radius: 0.375rem;
          font-size: 0.875rem;
          color: #d1d5db;
          word-break: break-all;
          font-family: 'Monaco', 'Courier New', monospace;
        }

        .deployment-result-copy-button {
          padding: 0.5rem 0.75rem;
          background: #2563eb;
          color: white;
          border: none;
          border-radius: 0.375rem;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.3s ease;
          white-space: nowrap;
          font-family: inherit;
        }

        .deployment-result-copy-button:hover {
          background: #1d4ed8;
        }

        .deployment-result-url-card {
          padding: 1rem;
          border-radius: 0.5rem;
          background: #1f2937;
          border: 1px solid #4b5563;
        }

        .deployment-result-endpoints {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .deployment-result-endpoints-title {
          font-weight: 600;
          color: #ffffff;
          margin: 0;
        }

        .deployment-result-endpoint-item {
          padding: 1rem;
          border-radius: 0.5rem;
          background: #1f2937;
          border: 1px solid #4b5563;
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 0.5rem;
        }

        .deployment-result-endpoint-info {
          flex: 1;
        }

        .deployment-result-endpoint-name {
          font-size: 0.875rem;
          font-weight: 600;
          color: #9ca3af;
          margin: 0 0 0.25rem 0;
          text-transform: capitalize;
        }

        .deployment-result-endpoint-url {
          font-size: 0.75rem;
          color: #6b7280;
          word-break: break-all;
          margin: 0;
          font-family: 'Monaco', 'Courier New', monospace;
        }

        .deployment-result-usage {
          padding: 1rem;
          border-radius: 0.5rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid #3b82f6;
        }

        .deployment-result-usage-title {
          font-weight: 600;
          color: #60a5fa;
          margin: 0 0 0.75rem 0;
        }

        .deployment-result-usage-code {
          background: #111827;
          padding: 0.75rem;
          border-radius: 0.375rem;
          font-size: 0.75rem;
          color: #d1d5db;
          overflow-x: auto;
          margin: 0;
          font-family: 'Monaco', 'Courier New', monospace;
        }

        .deployment-result-actions {
          display: flex;
          gap: 0.75rem;
        }

        .deployment-result-action-button {
          flex: 1;
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          font-weight: 600;
          border: none;
          cursor: pointer;
          transition: all 0.3s ease;
          text-decoration: none;
          text-align: center;
          display: inline-block;
          font-family: inherit;
        }

        .deployment-result-action-button.primary {
          background: #2563eb;
          color: white;
        }

        .deployment-result-action-button.primary:hover {
          background: #1d4ed8;
        }

        .deployment-result-action-button.secondary {
          background: #374151;
          color: white;
        }

        .deployment-result-action-button.secondary:hover {
          background: #4b5563;
        }

        @media (max-width: 768px) {
          .deployment-result-grid {
            grid-template-columns: 1fr;
          }

          .deployment-result-actions {
            flex-direction: column;
          }
        }
      `}</style>

      <div className="deployment-result-wrapper">
        <div className="deployment-result-success">
          <h3 className="deployment-result-success-title">✅ Deployment Successful!</h3>
          <p className="deployment-result-success-text">Your AI model is now live and ready to use.</p>
        </div>

        <div className="deployment-result-grid">
          <div className="deployment-result-card">
            <label className="deployment-result-card-label">Sandbox ID</label>
            <div className="deployment-result-card-content">
              <code className="deployment-result-code">{sandboxId}</code>
              <button
                onClick={() => copyToClipboard(sandboxId, 'sandbox')}
                className="deployment-result-copy-button"
              >
                {copiedEndpoint === 'sandbox' ? '✓' : 'Copy'}
              </button>
            </div>
          </div>

          <div className="deployment-result-card">
            <label className="deployment-result-card-label">Model Type</label>
            <div className="deployment-result-code" style={{ textTransform: 'capitalize' }}>
              {modelType}
            </div>
          </div>
        </div>

        <div className="deployment-result-url-card">
          <label className="deployment-result-card-label">Deployment URL</label>
          <div className="deployment-result-card-content">
            <code className="deployment-result-code">{deploymentUrl}</code>
            <button
              onClick={() => copyToClipboard(deploymentUrl, 'url')}
              className="deployment-result-copy-button"
            >
              {copiedEndpoint === 'url' ? '✓' : 'Copy'}
            </button>
          </div>
        </div>

        <div className="deployment-result-endpoints">
          <h4 className="deployment-result-endpoints-title">API Endpoints</h4>
          {Object.entries(endpoints).map(([name, url]) => (
            <div key={name} className="deployment-result-endpoint-item">
              <div className="deployment-result-endpoint-info">
                <p className="deployment-result-endpoint-name">{name}</p>
                <code className="deployment-result-endpoint-url">{url}</code>
              </div>
              <button
                onClick={() => copyToClipboard(url, name)}
                className="deployment-result-copy-button"
              >
                {copiedEndpoint === name ? '✓' : 'Copy'}
              </button>
            </div>
          ))}
        </div>

        <div className="deployment-result-usage">
          <h4 className="deployment-result-usage-title">📝 Usage Example</h4>
          <pre className="deployment-result-usage-code">{`curl -X POST ${endpoints.predict} \\
  -H "Content-Type: application/json" \\
  -d '{"input": [1.0, 2.0, 3.0]}'`}</pre>
        </div>

        <div className="deployment-result-actions">
          <a
            href={endpoints.health}
            target="_blank"
            rel="noopener noreferrer"
            className="deployment-result-action-button primary"
          >
            Test Health Endpoint
          </a>
          <button
            onClick={() => window.location.reload()}
            className="deployment-result-action-button secondary"
          >
            Create Another Model
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '100%',
  },
};
