'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import styles from './generation.module.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface TrainingStats {
  epoch?: number;
  totalEpochs?: number;
  loss?: number;
  accuracy?: number;
  eta?: string;
}

export default function GenerationPage() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showCodeView, setShowCodeView] = useState(false);
  const [sandboxOutput, setSandboxOutput] = useState('');
  const [trainingStats, setTrainingStats] = useState<TrainingStats>({});
  const [modelUrl, setModelUrl] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [importGuidelines, setImportGuidelines] = useState('');

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputMessage,
          userId: user?.id,
        }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = async () => {
    await signOut();
    router.push('/login');
  };

  if (authLoading) {
    return (
      <div className={styles.dashboard}>
        <div className={styles.loadingContainer}>
          <div className={styles.spinner}></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.dashboard}>
      {/* Header */}
      <div className={styles.header}>
        <h1>Model Generation</h1>
        <div className={styles.headerActions}>
          <button 
            className={`${styles.viewToggle} ${showCodeView ? styles.active : ''}`}
            onClick={() => setShowCodeView(!showCodeView)}
          >
            {showCodeView ? '🖥️ Code View' : '📦 Sandbox View'}
          </button>
          <span className={styles.userEmail}>{user?.email}</span>
          <button className={styles.signOutBtn} onClick={handleSignOut}>Sign Out</button>
        </div>
      </div>

      <div className={styles.container}>
        {/* Left Sidebar - AI Assistant */}
        <div className={styles.sidebar}>
          <div className={styles.sidebarHeader}>
            <h2>AI Assistant</h2>
          </div>

          {/* Messages */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <div className={styles.emptyState}>
                <p>👋 Hi! I'm your AI assistant. Ask me anything about your model or how to use it.</p>
              </div>
            ) : (
              messages.map(msg => (
                <div key={msg.id} className={`${styles.message} ${styles[msg.role]}`}>
                  <div className={styles.messageContent}>{msg.content}</div>
                  <div className={styles.messageTime}>
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSendMessage} className={styles.inputForm}>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask AI assistant..."
              className={styles.input}
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className={styles.sendBtn}
              disabled={isLoading || !inputMessage.trim()}
            >
              ➤
            </button>
          </form>
        </div>

        {/* Right Content - Sandbox/Code View */}
        <div className={styles.content}>
          {!isComplete ? (
            <div className={styles.generationView}>
              <div className={styles.statsCard}>
                <h3>Training Progress</h3>
                {trainingStats.epoch && (
                  <div className={styles.statItem}>
                    <span>Epoch:</span>
                    <strong>{trainingStats.epoch}/{trainingStats.totalEpochs}</strong>
                  </div>
                )}
                {trainingStats.loss && (
                  <div className={styles.statItem}>
                    <span>Loss:</span>
                    <strong>{trainingStats.loss.toFixed(4)}</strong>
                  </div>
                )}
                {trainingStats.accuracy && (
                  <div className={styles.statItem}>
                    <span>Accuracy:</span>
                    <strong>{(trainingStats.accuracy * 100).toFixed(2)}%</strong>
                  </div>
                )}
                {trainingStats.eta && (
                  <div className={styles.statItem}>
                    <span>ETA:</span>
                    <strong>{trainingStats.eta}</strong>
                  </div>
                )}
              </div>

              <div className={styles.outputCard}>
                <h3>Sandbox Output</h3>
                <pre className={styles.output}>{sandboxOutput || 'Waiting for output...'}</pre>
              </div>
            </div>
          ) : (
            <div className={styles.completionView}>
              <div className={styles.successMessage}>
                <h2>✅ Model Training Complete!</h2>
                <p>Your model has been successfully trained and deployed.</p>
              </div>

              <div className={styles.urlCard}>
                <h3>Model URL</h3>
                <code className={styles.url}>{modelUrl}</code>
                <button 
                  className={styles.copyBtn}
                  onClick={() => navigator.clipboard.writeText(modelUrl)}
                >
                  📋 Copy
                </button>
              </div>

              <div className={styles.guidelinesCard}>
                <h3>How to Use Your Model</h3>
                <pre className={styles.guidelines}>{importGuidelines}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
