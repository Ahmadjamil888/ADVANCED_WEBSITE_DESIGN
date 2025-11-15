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
    <div className="w-full space-y-4">
      <h3 className="text-lg font-semibold text-white mb-4">Training Progress</h3>

      <div className="space-y-3">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border-2 transition-all ${
              step.status === 'completed'
                ? 'border-green-500 bg-green-500/10'
                : step.status === 'error'
                ? 'border-red-500 bg-red-500/10'
                : step.status === 'in-progress'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-600 bg-gray-800'
            }`}
          >
            <div className="flex items-start gap-3">
              <span className="text-2xl mt-1">
                {STEP_ICONS[step.status]}
              </span>
              <div className="flex-1">
                <h4 className="font-semibold text-white">{step.name}</h4>
                {step.details && (
                  <p className="text-sm text-gray-300 mt-1">{step.details}</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="p-4 rounded-lg border-2 border-red-500 bg-red-500/10">
          <h4 className="font-semibold text-red-400 mb-2">Error</h4>
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}
    </div>
  );
}
