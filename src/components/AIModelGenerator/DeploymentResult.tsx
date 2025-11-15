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
    <div className="w-full space-y-6">
      <div className="p-6 rounded-lg bg-green-500/10 border-2 border-green-500">
        <h3 className="text-2xl font-bold text-green-400 mb-2">✅ Deployment Successful!</h3>
        <p className="text-gray-300">Your AI model is now live and ready to use.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-lg bg-gray-800 border border-gray-600">
          <label className="block text-sm font-semibold text-gray-300 mb-2">Sandbox ID</label>
          <div className="flex items-center gap-2">
            <code className="flex-1 p-2 bg-gray-900 rounded text-sm text-gray-200 break-all">
              {sandboxId}
            </code>
            <button
              onClick={() => copyToClipboard(sandboxId, 'sandbox')}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm"
            >
              {copiedEndpoint === 'sandbox' ? '✓' : 'Copy'}
            </button>
          </div>
        </div>

        <div className="p-4 rounded-lg bg-gray-800 border border-gray-600">
          <label className="block text-sm font-semibold text-gray-300 mb-2">Model Type</label>
          <div className="p-2 bg-gray-900 rounded text-sm text-gray-200 capitalize">
            {modelType}
          </div>
        </div>
      </div>

      <div className="p-4 rounded-lg bg-gray-800 border border-gray-600">
        <label className="block text-sm font-semibold text-gray-300 mb-2">Deployment URL</label>
        <div className="flex items-center gap-2">
          <code className="flex-1 p-2 bg-gray-900 rounded text-sm text-gray-200 break-all">
            {deploymentUrl}
          </code>
          <button
            onClick={() => copyToClipboard(deploymentUrl, 'url')}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm whitespace-nowrap"
          >
            {copiedEndpoint === 'url' ? '✓' : 'Copy'}
          </button>
        </div>
      </div>

      <div className="space-y-3">
        <h4 className="font-semibold text-white">API Endpoints</h4>
        {Object.entries(endpoints).map(([name, url]) => (
          <div key={name} className="p-4 rounded-lg bg-gray-800 border border-gray-600">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1">
                <p className="text-sm font-semibold text-gray-300 capitalize mb-1">{name}</p>
                <code className="text-xs text-gray-400 break-all">{url}</code>
              </div>
              <button
                onClick={() => copyToClipboard(url, name)}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm whitespace-nowrap"
              >
                {copiedEndpoint === name ? '✓' : 'Copy'}
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500">
        <h4 className="font-semibold text-blue-400 mb-2">📝 Usage Example</h4>
        <pre className="bg-gray-900 p-3 rounded text-xs text-gray-300 overflow-x-auto">
{`curl -X POST ${endpoints.predict} \\
  -H "Content-Type: application/json" \\
  -d '{"input": [1.0, 2.0, 3.0]}'`}
        </pre>
      </div>

      <div className="flex gap-3">
        <a
          href={endpoints.health}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-1 py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold text-center transition-all"
        >
          Test Health Endpoint
        </a>
        <button
          onClick={() => window.location.reload()}
          className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-all"
        >
          Create Another Model
        </button>
      </div>
    </div>
  );
}
