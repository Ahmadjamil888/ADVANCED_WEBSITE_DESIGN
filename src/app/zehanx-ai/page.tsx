'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function ZehanxAIDashboard() {
  const [stats, setStats] = useState({ totalModels: 0 });
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetch('/api/models');
        const data = await response.json();
        setStats({ totalModels: data.models?.length || 0 });
      } catch (error) {
        console.error('Failed to load stats:', error);
      }
    };
    loadStats();
  }, []);

  const handlePromptSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      const response = await fetch('/api/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Model_${Date.now()}`,
          description: prompt,
          modelType: 'custom',
          datasetSource: 'firecrawl',
          epochs: 10,
          batchSize: 32,
          learningRate: 0.001,
        }),
      });

      if (response.ok) {
        router.push('/zehanx-ai/generator');
      }
    } catch (error) {
      console.error('Error starting training:', error);
    } finally {
      setIsGenerating(false);
      setPrompt('');
    }
  };

  return (
    <>
      <style>{`
        .dashboard-container {
          width: 100%;
          height: 100%;
          background: #000;
          color: #fff;
          padding: 40px;
          overflow-y: auto;
        }

        .dashboard-header {
          margin-bottom: 40px;
        }

        .dashboard-title {
          font-size: 36px;
          font-weight: bold;
          margin-bottom: 10px;
          color: #fff;
        }

        .dashboard-subtitle {
          font-size: 16px;
          color: #888;
        }

        .prompt-section {
          background: #0a0a0a;
          border: 1px solid #222;
          border-radius: 12px;
          padding: 30px;
          margin-bottom: 40px;
        }

        .prompt-title {
          font-size: 20px;
          font-weight: bold;
          margin-bottom: 10px;
          color: #fff;
        }

        .prompt-description {
          font-size: 14px;
          color: #888;
          margin-bottom: 20px;
        }

        .prompt-form {
          display: flex;
          gap: 12px;
        }

        .prompt-input {
          flex: 1;
          padding: 12px 16px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          color: #fff;
          font-size: 14px;
          transition: all 0.2s;
        }

        .prompt-input:focus {
          outline: none;
          border-color: #fff;
          background: #222;
        }

        .prompt-input::placeholder {
          color: #666;
        }

        .prompt-button {
          padding: 12px 24px;
          background: #fff;
          color: #000;
          border: none;
          border-radius: 8px;
          font-weight: bold;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .prompt-button:hover:not(:disabled) {
          background: #ddd;
          transform: translateY(-2px);
        }

        .prompt-button:disabled {
          background: #444;
          color: #888;
          cursor: not-allowed;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin-bottom: 40px;
        }

        .stat-card {
          background: #0a0a0a;
          border: 1px solid #222;
          border-radius: 12px;
          padding: 24px;
          transition: all 0.2s;
        }

        .stat-card:hover {
          border-color: #444;
          background: #111;
        }

        .stat-label {
          font-size: 12px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 12px;
        }

        .stat-value {
          font-size: 32px;
          font-weight: bold;
          color: #fff;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 20px;
          margin-bottom: 40px;
        }

        .feature-card {
          background: #0a0a0a;
          border: 1px solid #222;
          border-radius: 12px;
          padding: 24px;
          cursor: pointer;
          transition: all 0.2s;
          text-decoration: none;
          color: inherit;
          display: block;
        }

        .feature-card:hover {
          border-color: #fff;
          background: #111;
          transform: translateY(-4px);
        }

        .feature-icon {
          font-size: 32px;
          margin-bottom: 12px;
        }

        .feature-title {
          font-size: 18px;
          font-weight: bold;
          margin-bottom: 8px;
          color: #fff;
        }

        .feature-description {
          font-size: 14px;
          color: #888;
          line-height: 1.5;
        }

        .quickstart-section {
          background: #0a0a0a;
          border: 1px solid #222;
          border-radius: 12px;
          padding: 30px;
          margin-bottom: 40px;
        }

        .quickstart-title {
          font-size: 20px;
          font-weight: bold;
          margin-bottom: 20px;
          color: #fff;
        }

        .quickstart-list {
          list-style: none;
          counter-reset: step-counter;
        }

        .quickstart-item {
          counter-increment: step-counter;
          display: flex;
          gap: 16px;
          margin-bottom: 16px;
          font-size: 14px;
          color: #ccc;
        }

        .quickstart-number {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          background: #222;
          border-radius: 50%;
          font-weight: bold;
          color: #fff;
          flex-shrink: 0;
        }

        .features-list {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 16px;
        }

        .feature-item {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          color: #ccc;
        }

        .feature-check {
          color: #fff;
          font-weight: bold;
        }
      `}</style>

      <div className="dashboard-container">
        <div className="dashboard-header">
          <h1 className="dashboard-title">Zehanx AI Dashboard</h1>
          <p className="dashboard-subtitle">Create, train, and deploy custom AI models</p>
        </div>

        <form onSubmit={handlePromptSubmit} className="prompt-section">
          <h2 className="prompt-title">Create AI Model</h2>
          <p className="prompt-description">
            Describe the AI model you want to create. Our system will crawl datasets, generate code with Groq, and train your model.
          </p>
          <div className="prompt-form">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Create a sentiment analysis model for Twitter data..."
              className="prompt-input"
              disabled={isGenerating}
            />
            <button
              type="submit"
              disabled={isGenerating || !prompt.trim()}
              className="prompt-button"
            >
              {isGenerating ? 'Starting...' : 'Generate'}
            </button>
          </div>
        </form>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total Models</div>
            <div className="stat-value">{stats.totalModels}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Datasets Available</div>
            <div className="stat-value">4</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Training Status</div>
            <div className="stat-value">Ready</div>
          </div>
        </div>

        <div className="features-grid">
          <a href="/zehanx-ai/generator" className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3 className="feature-title">Model Generator</h3>
            <p className="feature-description">Create and train custom AI models with multiple architectures</p>
          </a>

          <a href="/zehanx-ai/datasets" className="feature-card">
            <div className="feature-icon">📚</div>
            <h3 className="feature-title">Datasets</h3>
            <p className="feature-description">Manage and explore your training datasets from multiple sources</p>
          </a>

          <a href="/zehanx-ai/models" className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3 className="feature-title">My Models</h3>
            <p className="feature-description">View, download, and manage your trained AI models</p>
          </a>
        </div>

        <div className="quickstart-section">
          <h2 className="quickstart-title">Quick Start Guide</h2>
          <ol className="quickstart-list">
            <li className="quickstart-item">
              <span className="quickstart-number">1</span>
              <span>Use the prompt box above to describe your AI model</span>
            </li>
            <li className="quickstart-item">
              <span className="quickstart-number">2</span>
              <span>System crawls datasets using Firecrawl</span>
            </li>
            <li className="quickstart-item">
              <span className="quickstart-number">3</span>
              <span>Groq generates optimized training code</span>
            </li>
            <li className="quickstart-item">
              <span className="quickstart-number">4</span>
              <span>E2B executes training and displays real-time stats</span>
            </li>
            <li className="quickstart-item">
              <span className="quickstart-number">5</span>
              <span>Download your trained model from My Models</span>
            </li>
          </ol>
        </div>

        <div className="quickstart-section">
          <h2 className="quickstart-title">Platform Features</h2>
          <div className="features-list">
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>Firecrawl Integration - Automatic dataset fetching</span>
            </div>
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>Groq API - Code generation for model training</span>
            </div>
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>E2B Sandbox - Secure code execution</span>
            </div>
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>Real-time Training Statistics - Live monitoring</span>
            </div>
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>Multiple Model Architectures - Transformer, LSTM, CNN</span>
            </div>
            <div className="feature-item">
              <span className="feature-check">✓</span>
              <span>Supabase Integration - User data persistence</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
