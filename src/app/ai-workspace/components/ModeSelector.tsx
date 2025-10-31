"use client";

import React, { useState } from 'react';

interface ModeSelectorProps {
  currentMode: string;
  onModeChange: (mode: string) => void;
}

const modes = [
  { id: 'chat', name: 'Chat', icon: '💬', description: 'Conversational AI chat' },
  { id: 'code', name: 'Code', icon: '💻', description: 'Code generation and editing' },
  { id: 'models', name: 'Models', icon: '🧠', description: 'AI model creation and management' },
  { id: 'fine-tune', name: 'Fine-tune', icon: '⚙️', description: 'Model training and fine-tuning' },
  { id: 'research', name: 'Research', icon: '🔬', description: 'Research and analysis' },
  { id: 'app-builder', name: 'App Builder', icon: '🏗️', description: 'Visual app builder' },
  { id: 'translate', name: 'Translate', icon: '🌐', description: 'Language translation' }
];

export default function ModeSelector({ currentMode, onModeChange }: ModeSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const currentModeData = modes.find(mode => mode.id === currentMode) || modes[0];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
        aria-label="Select mode"
      >
        <span className="text-sm">{currentModeData.icon}</span>
        <span className="text-sm font-medium text-gray-700">{currentModeData.name}</span>
        <svg 
          className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
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
          <div className="absolute top-full left-0 mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-20">
            <div className="p-2">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => {
                    onModeChange(mode.id);
                    setIsOpen(false);
                  }}
                  className={`w-full flex items-start space-x-3 p-3 rounded-lg transition-colors text-left ${
                    currentMode === mode.id 
                      ? 'bg-blue-50 border border-blue-200' 
                      : 'hover:bg-gray-50'
                  }`}
                >
                  <span className="text-lg flex-shrink-0 mt-0.5">{mode.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-gray-900">{mode.name}</div>
                    <div className="text-sm text-gray-500">{mode.description}</div>
                  </div>
                  {currentMode === mode.id && (
                    <svg className="w-4 h-4 text-blue-600 flex-shrink-0 mt-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}