'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import styles from './page-centered.module.css';

const AVAILABLE_MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', speed: 'Fast' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'OpenAI', speed: 'Very Fast' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', speed: 'Fast' },
  { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', speed: 'Very Fast' },
  { id: 'llama-2-70b', name: 'Llama 2 70B', provider: 'Meta', speed: 'Medium' },
];

export default function AIModelGeneratorPage() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();

  const [showModelModal, setShowModelModal] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [prompt, setPrompt] = useState('');
  const [customDataset, setCustomDataset] = useState<File | null>(null);
  const [customModel, setCustomModel] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [modelName, setModelName] = useState('');
  const [useAWS, setUseAWS] = useState(false);
  const [awsKey, setAwsKey] = useState('');
  const [error, setError] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);

  React.useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

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

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('modelKey', selectedModel);
      formData.append('userId', user?.id || '');
      if (customDataset) formData.append('dataset', customDataset);
      if (customModel) formData.append('model', customModel);
      formData.append('useAWS', useAWS.toString());
      if (useAWS) formData.append('awsKey', awsKey);

      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Generation failed');
      }

      // Navigate to generation interface
      router.push('/ai-model-generator/generation');
    } catch (err: any) {
      setError(err.message || 'An error occurred');
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
        <div className={styles.headerLeft}>
          <h1>AI Model Generator</h1>
        </div>
        <div className={styles.headerRight}>
          <button 
            className={styles.modelToggle}
            onClick={() => setShowModelModal(true)}
          >
            {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name || 'Select Model'}
          </button>
          <span className={styles.userEmail}>{user?.email}</span>
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

      {/* Center Container */}
      <div className={styles.centerContainer}>
        <div className={styles.promptBox}>
          <h2 className={styles.boxTitle}>Create your custom API</h2>
          
          <form onSubmit={handleSubmit} className={styles.form}>
            {/* Prompt */}
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

            {/* Dataset Upload */}
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
                  {datasetName || '📁 Click to upload dataset'}
                </label>
              </div>
            </div>

            {/* Model Upload */}
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
                  {modelName || '📁 Click to upload model'}
                </label>
              </div>
            </div>

            {/* AWS Toggle */}
            <div className={styles.toggleSection}>
              <label className={styles.toggleLabel}>
                <input 
                  type="checkbox"
                  checked={useAWS}
                  onChange={(e) => setUseAWS(e.target.checked)}
                  className={styles.checkbox}
                />
                Train with AWS (Optional)
              </label>
              {useAWS && (
                <input
                  type="password"
                  value={awsKey}
                  onChange={(e) => setAwsKey(e.target.value)}
                  placeholder="Enter your AWS API key"
                  className={styles.input}
                />
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
      </div>
    </div>
  );
}
