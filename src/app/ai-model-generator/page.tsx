'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import styles from './page-new.module.css';

const AVAILABLE_MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', speed: 'Fast' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'OpenAI', speed: 'Very Fast' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', speed: 'Fast' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', speed: 'Very Fast' },
  { id: 'llama-2-70b', name: 'Llama 2 70B', provider: 'Meta', speed: 'Medium' },
];

interface UsageData {
  tokensUsed: number;
  apisCreated: number;
  modelsDeployed: number;
  requestsThisMonth: number;
  costThisMonth: number;
}

interface Step {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  details?: string;
}

export default function AIModelGeneratorPage() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();
  
  const [activeTab, setActiveTab] = useState<'generator' | 'usage' | 'billing' | 'settings'>('generator');
  const [prompt, setPrompt] = useState('');
  const [customDataset, setCustomDataset] = useState<File | null>(null);
  const [customModel, setCustomModel] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [modelName, setModelName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showModelModal, setShowModelModal] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [steps, setSteps] = useState<Step[]>([
    { name: 'Code Generation', status: 'pending' },
    { name: 'Sandbox Creation', status: 'pending' },
    { name: 'Model Training', status: 'pending' },
    { name: 'E2B Deployment', status: 'pending' },
  ]);
  const [error, setError] = useState<string | undefined>(undefined);
  const [deploymentResult, setDeploymentResult] = useState<any>(null);
  const [usageData, setUsageData] = useState<UsageData>({
    tokensUsed: 0,
    apisCreated: 0,
    modelsDeployed: 0,
    requestsThisMonth: 0,
    costThisMonth: 0,
  });
  const [currentPlan, setCurrentPlan] = useState('Free');

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      fetchUsageData();
    }
  }, [user]);

  const fetchUsageData = async () => {
    try {
      const response = await fetch(`/api/usage?userId=${user?.id}`);
      if (response.ok) {
        const data = await response.json();
        setUsageData(data);
      }
    } catch (err) {
      console.error('Error fetching usage data:', err);
    }
  };

  const handleDatasetUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomDataset(file);
      setDatasetName(file.name);
    }
  };

  const handleModelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomModel(file);
      setModelName(file.name);
    }
  };

  const updateStep = (index: number, status: 'pending' | 'in-progress' | 'completed' | 'error', details?: string) => {
    setSteps((prev: Step[]) => {
      const newSteps = [...prev];
      newSteps[index] = { ...newSteps[index], status, details };
      return newSteps;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('Please enter a prompt');
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
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('userId', user?.id || '');
      if (customDataset) formData.append('dataset', customDataset);
      if (customModel) formData.append('model', customModel);

      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        body: formData,
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
              if (data.step !== undefined) {
                updateStep(data.step, data.status as any, data.details);
              }
              if (data.deploymentUrl) {
                setDeploymentResult(data);
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
        fullResponse = lines[lines.length - 1];
      }

      setIsLoading(false);
      fetchUsageData();
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
      <div 
        className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ''}`}
        onMouseEnter={() => setSidebarOpen(true)}
        onMouseLeave={() => setSidebarOpen(false)}
      >
        <div className={styles.sidebarContent}>
          <h3>Menu</h3>
          <nav className={styles.sidebarNav}>
            <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('generator'); setSidebarOpen(false); }}>Generator</a>
            <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('usage'); setSidebarOpen(false); }}>Usage</a>
            <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('billing'); setSidebarOpen(false); }}>Billing</a>
            <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('settings'); setSidebarOpen(false); }}>Settings</a>
          </nav>
          <button className={styles.signOutBtn} onClick={handleSignOut}>Sign Out</button>
        </div>
      </div>

      {/* Model Selection Modal */}
      {showModelModal && (
        <div className={styles.modalOverlay} onClick={() => setShowModelModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Select AI Model</h2>
              <button className={styles.modalClose} onClick={() => setShowModelModal(false)}>✕</button>
            </div>
            <div className={styles.modalContent}>
              {AVAILABLE_MODELS.map((model) => (
                <div 
                  key={model.id}
                  className={`${styles.modelOption} ${selectedModel === model.id ? styles.modelSelected : ''}`}
                  onClick={() => {
                    setSelectedModel(model.id);
                    setShowModelModal(false);
                  }}
                >
                  <div className={styles.modelInfo}>
                    <div className={styles.modelName}>{model.name}</div>
                    <div className={styles.modelProvider}>{model.provider} • {model.speed}</div>
                  </div>
                  {selectedModel === model.id && <div className={styles.modelCheckmark}>✓</div>}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className={styles.header}>
        <button 
          className={styles.menuToggle}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          ☰
        </button>
        <h1>AI Model Generator</h1>
        <div className={styles.headerActions}>
          <span className={styles.userEmail}>{user?.email}</span>
          <button className={styles.signOutBtn} onClick={handleSignOut}>Sign Out</button>
        </div>
      </div>

      <div className={styles.tabsContainer}>
        <button
          className={`${styles.tab} ${activeTab === 'generator' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('generator')}
        >
          Generator
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'usage' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('usage')}
        >
          Usage
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'billing' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('billing')}
        >
          Billing
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'settings' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>

      {activeTab === 'generator' && (
        <div className={styles.content}>
          <div className={styles.generatorContainer}>
            <div className={styles.promptSection}>
              <div className={styles.promptHeader}>
                <h2>Create Your AI Model</h2>
                <button 
                  type="button"
                  className={styles.modelSelectorBtn}
                  onClick={() => setShowModelModal(true)}
                >
                  Model: {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name || 'Select'}
                </button>
              </div>
              <form onSubmit={handleSubmit} className={styles.form}>
                <div className={styles.formGroup}>
                  <label>Model Description</label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the AI model you want to create..."
                    className={styles.textarea}
                    rows={6}
                  />
                </div>

                <div className={styles.uploadSection}>
                  <h3>Upload Custom Dataset (Optional)</h3>
                  <div className={styles.uploadBox}>
                    <input
                      type="file"
                      accept=".csv,.json,.xlsx"
                      onChange={handleDatasetUpload}
                      className={styles.fileInput}
                      id="dataset-upload"
                    />
                    <label htmlFor="dataset-upload" className={styles.uploadLabel}>
                      {datasetName || 'Click to upload dataset'}
                    </label>
                  </div>
                </div>

                <div className={styles.uploadSection}>
                  <h3>Upload Custom Model (Optional)</h3>
                  <div className={styles.uploadBox}>
                    <input
                      type="file"
                      accept=".pth,.h5,.pb,.onnx,.safetensors"
                      onChange={handleModelUpload}
                      className={styles.fileInput}
                      id="model-upload"
                    />
                    <label htmlFor="model-upload" className={styles.uploadLabel}>
                      {modelName || 'Click to upload model'}
                    </label>
                  </div>
                </div>

                {error && <div className={styles.errorMessage}>{error}</div>}

                <button
                  type="submit"
                  disabled={isLoading || !prompt.trim()}
                  className={styles.submitButton}
                >
                  {isLoading ? 'Generating...' : 'Generate Model'}
                </button>
              </form>
            </div>

            <div className={styles.stepsSection}>
              <h2>Generation Progress</h2>
              <div className={styles.stepsContainer}>
                {steps.map((step, index) => (
                  <div key={index} className={`${styles.step} ${styles[step.status]}`}>
                    <div className={styles.stepNumber}>{index + 1}</div>
                    <div className={styles.stepContent}>
                      <div className={styles.stepName}>{step.name}</div>
                      {step.details && <div className={styles.stepDetails}>{step.details}</div>}
                    </div>
                    <div className={styles.stepStatus}>{step.status}</div>
                  </div>
                ))}
              </div>
            </div>

            {deploymentResult && (
              <div className={styles.resultSection}>
                <h2>Deployment Result</h2>
                <div className={styles.resultCard}>
                  <p>Deployment URL: <code>{deploymentResult.deploymentUrl}</code></p>
                  <p>Sandbox ID: <code>{deploymentResult.sandboxId}</code></p>
                  <button
                    onClick={() => window.open(deploymentResult.deploymentUrl, '_blank')}
                    className={styles.visitButton}
                  >
                    Visit Model
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'usage' && (
        <div className={styles.content}>
          <div className={styles.usageContainer}>
            <h2>Usage Statistics</h2>
            <div className={styles.usageGrid}>
              <div className={styles.usageCard}>
                <div className={styles.usageLabel}>Tokens Used</div>
                <div className={styles.usageValue}>{usageData.tokensUsed.toLocaleString()}</div>
              </div>
              <div className={styles.usageCard}>
                <div className={styles.usageLabel}>APIs Created</div>
                <div className={styles.usageValue}>{usageData.apisCreated}</div>
              </div>
              <div className={styles.usageCard}>
                <div className={styles.usageLabel}>Models Deployed</div>
                <div className={styles.usageValue}>{usageData.modelsDeployed}</div>
              </div>
              <div className={styles.usageCard}>
                <div className={styles.usageLabel}>Requests This Month</div>
                <div className={styles.usageValue}>{usageData.requestsThisMonth.toLocaleString()}</div>
              </div>
              <div className={styles.usageCard}>
                <div className={styles.usageLabel}>Cost This Month</div>
                <div className={styles.usageValue}>${usageData.costThisMonth.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'billing' && (
        <div className={styles.content}>
          <div className={styles.billingContainer}>
            <h2>Billing Plans</h2>
            <div className={styles.plansGrid}>
              <div className={`${styles.planCard} ${currentPlan === 'Free' ? styles.activePlan : ''}`}>
                <div className={styles.planName}>Free</div>
                <div className={styles.planPrice}>$0<span>/month</span></div>
                <ul className={styles.planFeatures}>
                  <li>1 AI Model</li>
                  <li>Basic Support</li>
                  <li>1,000 API Calls/month</li>
                </ul>
                <button className={styles.planButton} disabled={currentPlan === 'Free'}>
                  {currentPlan === 'Free' ? 'Current Plan' : 'Select'}
                </button>
              </div>

              <div className={`${styles.planCard} ${currentPlan === 'Pro' ? styles.activePlan : ''}`}>
                <div className={styles.planName}>Pro</div>
                <div className={styles.planPrice}>$80<span>/month</span></div>
                <ul className={styles.planFeatures}>
                  <li>10 AI Models</li>
                  <li>Priority Support</li>
                  <li>100,000 API Calls/month</li>
                </ul>
                <button className={styles.planButton} onClick={() => setCurrentPlan('Pro')}>
                  {currentPlan === 'Pro' ? 'Current Plan' : 'Upgrade'}
                </button>
              </div>

              <div className={`${styles.planCard} ${currentPlan === 'Enterprise' ? styles.activePlan : ''}`}>
                <div className={styles.planName}>Enterprise</div>
                <div className={styles.planPrice}>$100<span>/month</span></div>
                <ul className={styles.planFeatures}>
                  <li>Unlimited Models</li>
                  <li>24/7 Support</li>
                  <li>Unlimited API Calls</li>
                </ul>
                <button className={styles.planButton} onClick={() => setCurrentPlan('Enterprise')}>
                  {currentPlan === 'Enterprise' ? 'Current Plan' : 'Upgrade'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className={styles.content}>
          <div className={styles.settingsContainer}>
            <h2>Settings</h2>
            <div className={styles.settingItem}>
              <label>Email</label>
              <input type="email" value={user?.email || ''} disabled className={styles.settingInput} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
