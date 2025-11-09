'use client';

import { AI_MODELS } from '@/lib/ai/models';
import { useState } from 'react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelKey: string) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const currentModel = AI_MODELS[selectedModel];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg border border-gray-700 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="font-medium text-white">{currentModel.name}</span>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full mt-2 left-0 w-80 bg-gray-800 rounded-lg border border-gray-700 shadow-xl z-20 overflow-hidden">
            <div className="p-2">
              <div className="text-xs font-semibold text-gray-400 px-3 py-2">GROQ MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'groq')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                      selectedModel === key
                        ? 'bg-blue-600 text-white'
                        : 'hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    <div className="font-medium">{model.name}</div>
                    <div className="text-xs text-gray-400">{model.description}</div>
                  </button>
                ))}

              <div className="text-xs font-semibold text-gray-400 px-3 py-2 mt-2">GEMINI MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'gemini')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                      selectedModel === key
                        ? 'bg-blue-600 text-white'
                        : 'hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    <div className="font-medium">{model.name}</div>
                    <div className="text-xs text-gray-400">{model.description}</div>
                  </button>
                ))}

              <div className="text-xs font-semibold text-gray-400 px-3 py-2 mt-2">DEEPSEEK MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'deepseek')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                      selectedModel === key
                        ? 'bg-blue-600 text-white'
                        : 'hover:bg-gray-700 text-gray-300'
                    }`}
                  >
                    <div className="font-medium">{model.name}</div>
                    <div className="text-xs text-gray-400">{model.description}</div>
                  </button>
                ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
