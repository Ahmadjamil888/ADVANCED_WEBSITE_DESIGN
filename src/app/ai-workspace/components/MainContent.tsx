"use client";

import React, { useState, useRef, useEffect } from 'react';
import ModeSelector from './ModeSelector';
import MessageList from './MessageList';
import Composer from './Composer';

interface Chat {
  id: string;
  title: string;
  mode: string;
  updated_at: string;
  is_pinned: boolean;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  tokens_used?: number;
}

interface MainContentProps {
  currentChat: Chat | null;
  messages: Message[];
  currentMode: string;
  onModeChange: (mode: string) => void;
  onSendMessage: (content: string) => void;
  isLoading: boolean;
  contextPanelOpen: boolean;
  onToggleContextPanel: () => void;
}

export default function MainContent({
  currentChat,
  messages,
  currentMode,
  onModeChange,
  onSendMessage,
  isLoading,
  contextPanelOpen,
  onToggleContextPanel
}: MainContentProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  if (!currentChat) {
    return (
      <div className="flex-1 flex items-center justify-center bg-white">
        <div className="text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Welcome to AI Workspace</h3>
          <p className="text-gray-500">Create a new chat to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-lg font-semibold text-gray-900">{currentChat.title}</h1>
            <ModeSelector currentMode={currentMode} onModeChange={onModeChange} />
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={onToggleContextPanel}
              className={`p-2 rounded-lg transition-colors ${
                contextPanelOpen 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
              }`}
              aria-label="Toggle context panel"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2h2a2 2 0 002-2z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <img src="/logo.jpg" alt="AI" className="w-10 h-10 rounded-full" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Ready to create amazing AI models?
              </h3>
              <p className="text-gray-500 mb-6">
                I can help you generate, train, and deploy custom AI models. Just describe what you want to build!
              </p>
              
              {/* Quick Start Suggestions */}
              <div className="grid grid-cols-1 gap-3">
                <button
                  onClick={() => setInput("Create a text classification model for sentiment analysis")}
                  className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
                >
                  <div className="font-medium text-gray-900">🎯 Text Classification</div>
                  <div className="text-sm text-gray-500">Create a sentiment analysis model</div>
                </button>
                <button
                  onClick={() => setInput("Build an image classification model for detecting objects")}
                  className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
                >
                  <div className="font-medium text-gray-900">🖼️ Image Classification</div>
                  <div className="text-sm text-gray-500">Detect and classify objects in images</div>
                </button>
                <button
                  onClick={() => setInput("Generate a chatbot model for customer support")}
                  className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors"
                >
                  <div className="font-medium text-gray-900">🤖 Chatbot Model</div>
                  <div className="text-sm text-gray-500">Build a conversational AI assistant</div>
                </button>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} isLoading={isLoading} />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-gray-200">
        <Composer
          value={input}
          onChange={setInput}
          onSend={handleSend}
          isLoading={isLoading}
          mode={currentMode}
        />
      </div>
    </div>
  );
}