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
    <div className="w-full space-y-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-200 mb-2">
            Describe Your AI Model
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Create a sentiment analysis model that classifies text as positive, negative, or neutral..."
            className="w-full h-32 p-4 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none resize-none"
            disabled={isLoading}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !prompt.trim()}
          className={`w-full py-3 px-4 rounded-lg font-semibold transition-all ${
            isLoading || !prompt.trim()
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
          }`}
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="animate-spin">⚙️</span>
              Generating & Training Model...
            </span>
          ) : (
            'Generate & Train Model'
          )}
        </button>
      </form>

      {!isLoading && (
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Try an example:</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {EXAMPLE_PROMPTS.map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                className="p-3 text-left text-sm bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-gray-300 hover:text-white transition-all"
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
