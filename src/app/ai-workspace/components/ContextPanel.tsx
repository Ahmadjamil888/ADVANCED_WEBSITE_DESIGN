"use client";

import React, { useState } from 'react';

interface Chat {
  id: string;
  title: string;
  mode: string;
  updated_at: string;
  is_pinned: boolean;
}

interface ContextPanelProps {
  currentChat: Chat | null;
  onClose: () => void;
}

export default function ContextPanel({ currentChat, onClose }: ContextPanelProps) {
  const [activeTab, setActiveTab] = useState('context');

  const tabs = [
    { id: 'context', name: 'Context & Files', icon: '📁' },
    { id: 'entities', name: 'Entities & Memory', icon: '🧠' },
    { id: 'tools', name: 'Tools & Plugins', icon: '🔧' }
  ];

  const tools = [
    { 
      id: 'sql-runner', 
      name: 'SQL Runner', 
      icon: '🗃️', 
      enabled: true,
      description: 'Execute SQL queries and analyze data'
    },
    { 
      id: 'code-runner', 
      name: 'Code Runner', 
      icon: '▶️', 
      enabled: true,
      description: 'Run and test code in various languages'
    },
    { 
      id: 'api-playground', 
      name: 'API Playground', 
      icon: '🌐', 
      enabled: false,
      description: 'Test and debug API endpoints'
    },
    { 
      id: 'quiz-generator', 
      name: 'Quiz Generator', 
      icon: '📝', 
      enabled: true,
      description: 'Create educational quizzes and assessments'
    },
    { 
      id: 'model-trainer', 
      name: 'Model Trainer', 
      icon: '🤖', 
      enabled: true,
      description: 'Train and fine-tune AI models'
    },
    { 
      id: 'dataset-finder', 
      name: 'Dataset Finder', 
      icon: '📊', 
      enabled: true,
      description: 'Search and import datasets from Kaggle/HF'
    }
  ];

  return (
    <div className="w-96 bg-gray-50 border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-gray-900">Context Panel</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 rounded"
            aria-label="Close context panel"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mt-4 bg-gray-200 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <span>{tab.icon}</span>
              <span className="hidden lg:inline">{tab.name.split(' ')[0]}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'context' && (
          <div className="space-y-4">
            <div>
              <h3 className="font-medium text-gray-900 mb-3">Files & Context</h3>
              
              {/* Upload Area */}
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors cursor-pointer">
                <svg className="w-8 h-8 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-sm text-gray-600">Drop files here or click to upload</p>
                <p className="text-xs text-gray-500 mt-1">PDF, TXT, CSV, JSON supported</p>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Vector Database</h4>
              <div className="bg-white rounded-lg border border-gray-200 p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">Index Status</span>
                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">Ready</span>
                </div>
                <div className="text-xs text-gray-500">
                  0 documents indexed • 0 embeddings
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'entities' && (
          <div className="space-y-4">
            <div>
              <h3 className="font-medium text-gray-900 mb-3">Conversation Memory</h3>
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="text-center text-gray-500">
                  <svg className="w-8 h-8 mx-auto mb-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <p className="text-sm">No entities detected yet</p>
                  <p className="text-xs text-gray-400 mt-1">Start a conversation to see extracted entities</p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Key Concepts</h4>
              <div className="space-y-2">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <div className="font-medium text-blue-900 text-sm">AI Model Training</div>
                  <div className="text-xs text-blue-700 mt-1">Mentioned 0 times</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="space-y-4">
            <div>
              <h3 className="font-medium text-gray-900 mb-3">Available Tools</h3>
              <div className="space-y-3">
                {tools.map((tool) => (
                  <div key={tool.id} className="bg-white rounded-lg border border-gray-200 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{tool.icon}</span>
                        <span className="font-medium text-gray-900 text-sm">{tool.name}</span>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={tool.enabled}
                          className="sr-only peer"
                          onChange={() => {
                            // TODO: Toggle tool
                          }}
                        />
                        <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>
                    <p className="text-xs text-gray-500">{tool.description}</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Quick Actions</h4>
              <div className="space-y-2">
                <button className="w-full text-left p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-center space-x-2">
                    <span>🚀</span>
                    <span className="text-sm font-medium">Deploy to Hugging Face</span>
                  </div>
                </button>
                <button className="w-full text-left p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-center space-x-2">
                    <span>📊</span>
                    <span className="text-sm font-medium">Find Dataset</span>
                  </div>
                </button>
                <button className="w-full text-left p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-center space-x-2">
                    <span>⚡</span>
                    <span className="text-sm font-medium">Run in Sandbox</span>
                  </div>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}