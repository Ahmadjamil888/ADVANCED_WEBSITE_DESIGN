'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import styles from './page.module.css';

// Groq supported models
const GROQ_MODELS = [
  { id: 'llama-3.1-8b-instant', name: 'Llama 3.1 8B (Fastest)', speed: '560 T/sec' },
  { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B (Balanced)', speed: '280 T/sec' },
  { id: 'openai/gpt-oss-120b', name: 'GPT OSS 120B (Most Powerful)', speed: '500 T/sec' },
  { id: 'openai/gpt-oss-20b', name: 'GPT OSS 20B', speed: '1000 T/sec' },
  { id: 'meta-llama/llama-guard-4-12b', name: 'Llama Guard 4 (Safety)', speed: '1200 T/sec' },
  { id: 'groq/compound', name: 'Groq Compound (System)', speed: '450 T/sec' },
  { id: 'groq/compound-mini', name: 'Groq Compound Mini', speed: '450 T/sec' },
];

interface Step {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  details?: string;
}

export default function AIModelGeneratorPageV2() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama-3.3-70b-versatile');
  const [useAWS, setUseAWS] = useState(false);
  const [awsKey, setAwsKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [steps, setSteps] = useState<Step[]>([
    { name: 'Code Generation', status: 'pending' },
    { name: 'Sandbox Creation', status: 'pending' },
    { name: 'Model Training', status: 'pending' },
    { name: 'E2B Deployment', status: 'pending' },
  ]);
  const [error, setError] = useState<string | undefined>(undefined);
  const [deploymentResult, setDeploymentResult] = useState<any>(null);
  const [modelDocs, setModelDocs] = useState<string>('');

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  const updateStep = (index: number, status: 'pending' | 'in-progress' | 'completed' | 'error', details?: string) => {
    setSteps((prev: Step[]) => {
      const newSteps = [...prev];
      newSteps[index] = { ...newSteps[index], status, details };
      return newSteps;
    });
  };

  const generateModelDocs = (modelId: string) => {
    const modelInfo = GROQ_MODELS.find(m => m.id === modelId);
    if (!modelInfo) return '';

    return `
# ${modelInfo.name}

## Model Information
- **Model ID**: ${modelId}
- **Speed**: ${modelInfo.speed}
- **Provider**: Groq Cloud

## Quick Start
\`\`\`python
from groq import Groq

client = Groq(api_key="your-groq-api-key")
response = client.chat.completions.create(
    model="${modelId}",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
\`\`\`

## Features
- High-speed inference
- Optimized for production
- OpenAI-compatible API
- Streaming support

## Use Cases
- Real-time AI applications
- Code generation
- Data analysis
- Content creation
- Chat applications

## Integration with E2B
This model will be deployed to E2B sandbox on port 49999 for secure execution.

## API Endpoints
- **Health**: GET /health
- **Info**: GET /info
- **Predict**: POST /predict

## Rate Limits
- Developer Plan: 250K TPM, 1K RPM
- Production: Higher limits available

## Documentation
For more details, visit: https://console.groq.com/docs
    `;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    if (useAWS && !awsKey.trim()) {
      setError('AWS key is required when AWS training is enabled');
      return;
    }

    setIsLoading(true);
    setError(undefined);
    setDeploymentResult(null);
    setSteps([
      { name: 'Code Generation', status: 'in-progress' },
      { name: 'Sandbox Creation', status: 'pending' },
      { name: 'Model Training', status: 'pending' },
      { name: 'E2B Deployment', status: 'pending' },
    ]);

    try {
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          modelKey: selectedModel,
          userId: user?.id,
          useAWS,
          awsKey: useAWS ? awsKey : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error('Generation failed');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        fullResponse += decoder.decode(value);
        const lines = fullResponse.split('\n');

        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'status' && data.data.step) {
                updateStep(data.data.step - 1, 'in-progress', data.data.message);
              }
              if (data.type === 'deployment-url') {
                setDeploymentResult({ deploymentUrl: data.data.url });
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
        fullResponse = lines[lines.length - 1];
      }

      setIsLoading(false);
      setModelDocs(generateModelDocs(selectedModel));
    } catch (err: any) {
      setError(err.message || 'An error occurred');
      updateStep(0, 'error', err.message);
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
      {/* Sidebar */}
      <div className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ''}`}>
        <div className={styles.sidebarHeader}>
          <h3>Menu</h3>
          <button 
            className={styles.closeBtn}
            onClick={() => setSidebarOpen(false)}
          >
            ✕
          </button>
        </div>
        <nav className={styles.sidebarNav}>
          <a href="#" className={styles.navItem}>Dashboard</a>
          <a href="#" className={styles.navItem}>Models</a>
          <a href="#" className={styles.navItem}>Usage</a>
          <a href="#" className={styles.navItem}>Billing</a>
          <a href="#" className={styles.navItem}>Settings</a>
        </nav>
        <div className={styles.sidebarFooter}>
          <button className={styles.signOutBtn} onClick={handleSignOut}>
            Sign Out
          </button>
        </div>
      </div>

      {/* Overlay */}
      {sidebarOpen && (
        <div 
          className={styles.overlay}
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}

      {/* Main Content */}
      <div className={styles.mainContent}>
        {/* Header */}
        <div className={styles.header}>
          <button 
            className={styles.menuBtn}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <h1>AI Model Generator</h1>
          <span className={styles.userEmail}>{user?.email}</span>
        </div>

        {/* Center Container */}
        <div className={styles.centerContainer}>
          {/* Prompt Box */}
          <div className={styles.promptBox}>
            <h2>Create Your AI Model</h2>
            <form onSubmit={handleSubmit} className={styles.form}>
              {/* Model Selector */}
              <div className={styles.formGroup}>
                <label>Select Groq Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setModelDocs(generateModelDocs(e.target.value));
                  }}
                  className={styles.select}
                >
                  {GROQ_MODELS.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} - {model.speed}
                    </option>
                  ))}
                </select>
              </div>

              {/* Prompt Textarea */}
              <div className={styles.formGroup}>
                <label>Model Description</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the AI model you want to create..."
                  className={styles.textarea}
                  rows={8}
                />
              </div>

              {/* AWS Training Toggle */}
              <div className={styles.toggleSection}>
                <div className={styles.toggleHeader}>
                  <label className={styles.toggleLabel}>
                    <input 
                      type="checkbox"
                      checked={useAWS}
                      onChange={(e) => setUseAWS(e.target.checked)}
                      className={styles.checkbox}
                    />
                    Train with AWS (Optional)
                  </label>
                  <span className={styles.toggleInfo}>
                    {useAWS ? 'AWS training enabled' : 'Using E2B sandbox'}
                  </span>
                </div>
                
                {useAWS && (
                  <div className={styles.awsKeyInput}>
                    <input
                      type="password"
                      value={awsKey}
                      onChange={(e) => setAwsKey(e.target.value)}
                      placeholder="Enter your AWS API key"
                      className={styles.input}
                    />
                  </div>
                )}
              </div>

              {/* Error Message */}
              {error && <div className={styles.errorMessage}>{error}</div>}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !prompt.trim()}
                className={styles.submitButton}
              >
                {isLoading ? 'Generating...' : 'Generate Model'}
              </button>
            </form>
          </div>

          {/* Steps Display */}
          {isLoading && (
            <div className={styles.stepsBox}>
              <h3>Generation Progress</h3>
              <div className={styles.stepsContainer}>
                {steps.map((step, index) => (
                  <div key={index} className={`${styles.step} ${styles[step.status]}`}>
                    <div className={styles.stepNumber}>{index + 1}</div>
                    <div className={styles.stepContent}>
                      <div className={styles.stepName}>{step.name}</div>
                      {step.details && <div className={styles.stepDetails}>{step.details}</div>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Deployment Result */}
          {deploymentResult && (
            <div className={styles.resultBox}>
              <h3>Deployment Successful</h3>
              <div className={styles.resultContent}>
                <p>Deployment URL: <code>{deploymentResult.deploymentUrl}</code></p>
                <button
                  onClick={() => window.open(deploymentResult.deploymentUrl, '_blank')}
                  className={styles.visitButton}
                >
                  Visit Model
                </button>
              </div>
            </div>
          )}

          {/* Model Documentation */}
          {modelDocs && (
            <div className={styles.docsBox}>
              <h3>Model Documentation</h3>
              <pre className={styles.docContent}>{modelDocs}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
