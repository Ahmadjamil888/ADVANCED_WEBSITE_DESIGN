'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelSelector } from './components/ModelSelector';
import { SandboxPreview } from './components/SandboxPreview';
import { ChatMessage } from './components/ChatMessage';
import { StatusIndicator } from './components/StatusIndicator';
import { DEFAULT_MODEL } from '@/lib/ai/models';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  files?: string[];
}

interface Status {
  message: string;
  step?: number;
  total?: number;
  type?: 'info' | 'success' | 'error' | 'warning';
}

export default function AIWorkspacePage() {
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStatus, setCurrentStatus] = useState<Status | null>(null);
  const [sandboxUrl, setSandboxUrl] = useState<string>();
  const [sandboxId, setSandboxId] = useState<string>();
  const [streamingContent, setStreamingContent] = useState('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStatus, streamingContent]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsGenerating(true);
    setStreamingContent('');
    setCurrentStatus(null);

    try {
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userMessage.content,
          modelKey: selectedModel,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader available');

      let fullResponse = '';
      let generatedFiles: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'status':
                setCurrentStatus({
                  message: data.data.message,
                  step: data.data.step,
                  total: data.data.total,
                  type: 'info',
                });
                break;

              case 'ai-stream':
                fullResponse += data.data.content;
                setStreamingContent(fullResponse);
                break;

              case 'files':
                generatedFiles = data.data.files;
                break;

              case 'sandbox':
                setSandboxId(data.data.sandboxId);
                break;

              case 'deployment-url':
                setSandboxUrl(data.data.url);
                break;

              case 'complete':
                setCurrentStatus({
                  message: data.data.message,
                  type: 'success',
                });
                setSandboxUrl(data.data.deploymentUrl);
                
                const assistantMessage: Message = {
                  id: Date.now().toString(),
                  role: 'assistant',
                  content: fullResponse || 'Model training completed successfully!',
                  files: data.data.files,
                };
                setMessages(prev => [...prev, assistantMessage]);
                setStreamingContent('');
                break;

              case 'error':
                setCurrentStatus({
                  message: data.data.message,
                  type: 'error',
                });
                break;
            }
          }
        }
      }
    } catch (error: any) {
      setCurrentStatus({
        message: error.message || 'An error occurred',
        type: 'error',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold text-white">AI Model Training Studio</h1>
          <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
        </div>
        <div className="flex items-center gap-3">
          <div className="text-sm text-gray-400">
            {sandboxId && (
              <span className="font-mono">
                Sandbox: {sandboxId.slice(0, 8)}...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Content - Split View */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Chat */}
        <div className="w-1/2 flex flex-col border-r border-gray-800">
          {/* Messages */}
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto px-6 py-6 space-y-4"
          >
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center max-w-md">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-bold text-white mb-2">
                    Start Training Your AI Model
                  </h2>
                  <p className="text-gray-400 mb-6">
                    Describe what you want to build, and I'll generate the code, train the model, and deploy it for you.
                  </p>
                  <div className="text-left bg-gray-900 rounded-lg p-4 space-y-2">
                    <div className="text-sm text-gray-500 mb-2">Try examples:</div>
                    <button
                      onClick={() => setInput('Create a sentiment analysis model using BERT for product reviews')}
                      className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-gray-300 transition-colors"
                    >
                      💬 Sentiment analysis with BERT
                    </button>
                    <button
                      onClick={() => setInput('Build an image classifier for cats vs dogs using ResNet')}
                      className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-gray-300 transition-colors"
                    >
                      🖼️ Image classification with ResNet
                    </button>
                    <button
                      onClick={() => setInput('Create a text generation model using GPT-2')}
                      className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-gray-300 transition-colors"
                    >
                      ✍️ Text generation with GPT-2
                    </button>
                  </div>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
                files={message.files}
              />
            ))}

            {streamingContent && (
              <ChatMessage
                role="assistant"
                content={streamingContent}
                isStreaming={true}
              />
            )}

            {currentStatus && (
              <StatusIndicator
                message={currentStatus.message}
                step={currentStatus.step}
                total={currentStatus.total}
                type={currentStatus.type}
              />
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-800 p-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Describe the AI model you want to train..."
                disabled={isGenerating}
                className="flex-1 bg-gray-800 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button
                type="submit"
                disabled={isGenerating || !input.trim()}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Generate
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Side - Sandbox Preview */}
        <div className="w-1/2 bg-gray-900">
          <SandboxPreview sandboxUrl={sandboxUrl} sandboxId={sandboxId} />
        </div>
      </div>
    </div>
  );
}
