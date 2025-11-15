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
    <div className="w-full">
      <label className="block text-sm font-semibold text-gray-200 mb-4">
        Select AI Model
      </label>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {AVAILABLE_MODELS.map((model) => (
          <button
            key={model.id}
            onClick={() => onModelChange(model.id)}
            className={`p-4 rounded-lg border-2 transition-all text-left ${
              selectedModel === model.id
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-600 bg-gray-800 hover:border-gray-500'
            }`}
          >
            <div className="flex items-start gap-3">
              <span className="text-2xl">{model.icon}</span>
              <div className="flex-1">
                <h3 className="font-semibold text-white">{model.name}</h3>
                <p className="text-sm text-gray-400">{model.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
