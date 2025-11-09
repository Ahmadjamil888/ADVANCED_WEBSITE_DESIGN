'use client';

import { useEffect, useState } from 'react';

interface SandboxPreviewProps {
  sandboxUrl?: string;
  sandboxId?: string;
}

export function SandboxPreview({ sandboxUrl, sandboxId }: SandboxPreviewProps) {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (sandboxUrl) {
      setIsLoading(true);
      // Give iframe time to load
      const timer = setTimeout(() => setIsLoading(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [sandboxUrl]);

  if (!sandboxUrl) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-800 flex items-center justify-center">
            <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-400 mb-2">No Sandbox Active</h3>
          <p className="text-sm text-gray-500">
            Start training a model to see the live preview here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full relative bg-gray-900">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-400">Loading sandbox...</p>
          </div>
        </div>
      )}
      
      <div className="h-full flex flex-col">
        <div className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
            </div>
            <div className="text-sm text-gray-400 font-mono">
              {sandboxId ? `Sandbox: ${sandboxId.slice(0, 8)}...` : 'Live Preview'}
            </div>
          </div>
          <a
            href={sandboxUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
          >
            Open in New Tab ↗
          </a>
        </div>
        
        <iframe
          src={sandboxUrl}
          className="w-full flex-1 bg-white"
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          onLoad={() => setIsLoading(false)}
        />
      </div>
    </div>
  );
}
