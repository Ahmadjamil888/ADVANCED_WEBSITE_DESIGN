'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelSelector } from './components/ModelSelector';
import { SandboxPreview } from './components/SandboxPreview';
import { ChatMessage } from './components/ChatMessage';
import { StatusIndicator } from './components/StatusIndicator';
import { DEFAULT_MODEL } from '@/lib/ai/models';
import styles from './page.module.css';

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
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>AI Model Training Studio</h1>
          <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
        </div>
        <div className={styles.headerRight}>
          <div className={styles.sandboxInfo}>
            {sandboxId && (
              <span>
                Sandbox: {sandboxId.slice(0, 8)}...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Content - Split View */}
      <div className={styles.mainContent}>
        {/* Left Sidebar - Chat */}
        <div className={styles.chatSidebar}>
          {/* Messages */}
          <div
            ref={chatContainerRef}
            className={styles.messagesContainer}
          >
            {messages.length === 0 && (
              <div className={styles.emptyState}>
                <div className={styles.emptyContent}>
                  <div className={styles.emptyIcon}>
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h2 className={styles.emptyTitle}>
                    Start Training Your AI Model
                  </h2>
                  <p className={styles.emptyDescription}>
                    Describe what you want to build, and I'll generate the code, train the model, and deploy it for you.
                  </p>
                  <div className={styles.examplesContainer}>
                    <div className={styles.examplesLabel}>Try examples:</div>
                    <button
                      onClick={() => setInput('Create a sentiment analysis model using BERT for product reviews')}
                      className={styles.exampleButton}
                    >
                      💬 Sentiment analysis with BERT
                    </button>
                    <button
                      onClick={() => setInput('Build an image classifier for cats vs dogs using ResNet')}
                      className={styles.exampleButton}
                    >
                      🖼️ Image classification with ResNet
                    </button>
                    <button
                      onClick={() => setInput('Create a text generation model using GPT-2')}
                      className={styles.exampleButton}
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
          <div className={styles.inputContainer}>
            <form onSubmit={handleSubmit} className={styles.inputForm}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Describe the AI model you want to train..."
                disabled={isGenerating}
                className={styles.input}
              />
              <button
                type="submit"
                disabled={isGenerating || !input.trim()}
                className={styles.submitButton}
              >
                {isGenerating ? (
                  <>
                    <div className={styles.spinner} />
                    Generating...
                  </>
                ) : (
                  <>
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
        <div className={styles.sandboxPanel}>
          <SandboxPreview sandboxUrl={sandboxUrl} sandboxId={sandboxId} />
        </div>
      </div>
    </div>
  );
}
