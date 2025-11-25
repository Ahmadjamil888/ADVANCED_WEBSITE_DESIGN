"use client";

import React, { useState, useRef } from 'react';

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  mode: string;
}

export default function Composer({ value, onChange, onSend, isLoading, mode }: ComposerProps) {
  const [showTemplates, setShowTemplates] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && value.trim()) {
        onSend();
      }
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  };

  const templates = [
    {
      name: "Text Classification Model",
      content: "Create a text classification model that can classify customer reviews as positive, negative, or neutral. Use a dataset from Hugging Face and fine-tune a BERT model."
    },
    {
      name: "Image Classification Model", 
      content: "Build an image classification model that can identify different types of animals. Find a suitable dataset from Kaggle and use a CNN architecture with PyTorch."
    },
    {
      name: "Chatbot Model",
      content: "Generate a conversational AI chatbot model for customer support. Train it on FAQ data and make it capable of handling common customer inquiries."
    },
    {
      name: "Time Series Prediction",
      content: "Create a time series forecasting model to predict stock prices or weather patterns. Use LSTM networks and historical data for training."
    }
  ];

  const actions = [
    { name: "Generate Tests", icon: "🧪", description: "Create unit tests for code" },
    { name: "Create Quiz", icon: "📝", description: "Generate educational quiz" },
    { name: "Create Agent", icon: "🤖", description: "Build AI agent" },
    { name: "Deploy Model", icon: "🚀", description: "Deploy to Hugging Face" }
  ];

  return (
    <div className="p-4">
      <div className="max-w-4xl mx-auto">
        {/* Composer Input */}
        <div className="relative bg-white border border-gray-300 rounded-2xl shadow-sm focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500">
          {/* Action Buttons Row */}
          <div className="flex items-center justify-between p-3 border-b border-gray-200">
            <div className="flex items-center space-x-2">
              {/* Attach Button */}
              <button
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                aria-label="Attach files"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.586-6.586a2 2 0 00-2.828-2.828z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4" />
                </svg>
              </button>

              {/* Snippet Button */}
              <button
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                aria-label="Code snippet"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </button>

              {/* Templates Button */}
              <div className="relative">
                <button
                  onClick={() => setShowTemplates(!showTemplates)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  aria-label="Templates"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </button>

                {/* Templates Dropdown */}
                {showTemplates && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setShowTemplates(false)} />
                    <div className="absolute bottom-full left-0 mb-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg z-20">
                      <div className="p-3 border-b border-gray-200">
                        <h3 className="font-medium text-gray-900">AI Model Templates</h3>
                      </div>
                      <div className="max-h-64 overflow-y-auto">
                        {templates.map((template, index) => (
                          <button
                            key={index}
                            onClick={() => {
                              onChange(template.content);
                              setShowTemplates(false);
                            }}
                            className="w-full p-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0"
                          >
                            <div className="font-medium text-gray-900 mb-1">{template.name}</div>
                            <div className="text-sm text-gray-500 line-clamp-2">{template.content}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Actions Button */}
              <button
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                aria-label="Actions"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </button>
            </div>

            {/* Mode indicator */}
            <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {mode === 'models' ? 'AI Model Generator' : mode.charAt(0).toUpperCase() + mode.slice(1)} Mode
            </div>
          </div>

          {/* Text Input */}
          <div className="p-4">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => {
                onChange(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder={
                mode === 'models' 
                  ? "Describe the AI model you want to create (e.g., 'Create a sentiment analysis model using BERT...')"
                  : "Type your message here..."
              }
              className="w-full resize-none border-none outline-none text-gray-900 placeholder-gray-500"
              style={{ minHeight: '60px', maxHeight: '200px' }}
              disabled={isLoading}
            />
          </div>

          {/* Send Button */}
          <div className="flex justify-end p-3 pt-0">
            <button
              onClick={onSend}
              disabled={!value.trim() || isLoading}
              className={`p-3 rounded-xl transition-all ${
                value.trim() && !isLoading
                  ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
              aria-label="Send message"
            >
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="text-center mt-3">
          <p className="text-xs text-gray-500">
            zehanx AI can generate, train, and deploy custom AI models. Always verify generated code before training.
          </p>
        </div>
      </div>
    </div>
  );
}