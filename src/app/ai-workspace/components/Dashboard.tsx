'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/lib/supabase';
import { SignOutButton } from './SignOutButton';
import styles from './Dashboard.module.css';

interface AIModel {
  id: string;
  name: string;
  description: string;
  model_type: string;
  training_status: string;
  performance_metrics: any;
  created_at: string;
}

interface Billing {
  plan_type: 'free' | 'pro' | 'enterprise';
  models_created: number;
  models_limit: number;
  has_api_access: boolean;
}

export function Dashboard() {
  const { user, loading } = useAuth();
  const [models, setModels] = useState<AIModel[]>([]);
  const [billing, setBilling] = useState<Billing | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<'models' | 'datasets' | 'trained' | 'in_progress' | 'billing'>('models');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [showCreateModal, setShowCreateModal] = useState(false);

  useEffect(() => {
    if (!loading && user) {
      loadModels();
      loadBilling();
    }
  }, [loading, user]);

  useEffect(() => {
    const savedTheme = localStorage.getItem('ai-workspace-theme') as 'dark' | 'light' | null;
    if (savedTheme) setTheme(savedTheme);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-ai-theme', theme);
    localStorage.setItem('ai-workspace-theme', theme);
  }, [theme]);

  const loadModels = async () => {
    if (!user || !supabase) return;
    try {
      const { data, error } = await (supabase
        .from('ai_models')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false }) as any)();
      if (!error && data) setModels(data);
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const loadBilling = async () => {
    if (!user || !supabase) return;
    try {
      const { data, error } = await (supabase
        .from('billing')
        .select('*')
        .eq('user_id', user.id)
        .single() as any)();
      if (!error && data) setBilling(data);
    } catch (err) {
      console.error('Error loading billing:', err);
    }
  };

  const canCreateModel = () => {
    if (!billing) return false;
    return billing.models_created < billing.models_limit;
  };

  if (loading) {
    return (
      <div className={styles.dashboard} data-theme={theme}>
        <div className={styles.loading}>Loading...</div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className={styles.dashboard} data-theme={theme}>
      {/* Top Bar */}
      <div className={styles.topBar}>
        <div className={styles.topBarLeft}>
          <button
            className={styles.sidebarToggle}
            onClick={() => setSidebarOpen(!sidebarOpen)}
            onMouseEnter={() => setSidebarOpen(true)}
          >
            ☰
          </button>
          <h1 className={styles.brand}>zehanxtech</h1>
        </div>
        <div className={styles.topBarRight}>
          <button
            className={styles.themeToggle}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <SignOutButton />
        </div>
      </div>

      <div className={styles.mainLayout}>
        {/* Sidebar */}
        <div
          className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ''}`}
          onMouseEnter={() => setSidebarOpen(true)}
          onMouseLeave={() => setSidebarOpen(false)}
        >
          <div className={styles.sidebarContent}>
            <button
              className={`${styles.sidebarItem} ${activeSection === 'models' ? styles.active : ''}`}
              onClick={() => setActiveSection('models')}
            >
              <span>🤖</span>
              <span>LLMs</span>
            </button>
            <button
              className={`${styles.sidebarItem} ${activeSection === 'datasets' ? styles.active : ''}`}
              onClick={() => setActiveSection('datasets')}
            >
              <span>📊</span>
              <span>Datasets</span>
            </button>
            <button
              className={`${styles.sidebarItem} ${activeSection === 'trained' ? styles.active : ''}`}
              onClick={() => setActiveSection('trained')}
            >
              <span>✅</span>
              <span>Trained Models</span>
            </button>
            <button
              className={`${styles.sidebarItem} ${activeSection === 'in_progress' ? styles.active : ''}`}
              onClick={() => setActiveSection('in_progress')}
            >
              <span>⚙️</span>
              <span>In Progress</span>
            </button>
            <button
              className={`${styles.sidebarItem} ${activeSection === 'billing' ? styles.active : ''}`}
              onClick={() => setActiveSection('billing')}
            >
              <span>💳</span>
              <span>Billing</span>
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className={styles.content}>
          {activeSection === 'models' && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h2>AI Models</h2>
                <button
                  className={styles.createButton}
                  onClick={() => setShowCreateModal(true)}
                  disabled={!canCreateModel()}
                >
                  + Create AI Model
                </button>
              </div>
              {!canCreateModel() && (
                <div className={styles.warning}>
                  You've reached your model limit ({billing?.models_limit}). Upgrade your plan to create more models.
                </div>
              )}
              <div className={styles.modelsGrid}>
                {models.length === 0 ? (
                  <div className={styles.emptyState}>
                    <p>No models yet. Create your first AI model!</p>
                  </div>
                ) : (
                  models.map((model) => (
                    <div key={model.id} className={styles.modelCard}>
                      <h3>{model.name}</h3>
                      <p>{model.description || 'No description'}</p>
                      <div className={styles.modelMeta}>
                        <span>Status: {model.training_status}</span>
                        <span>Type: {model.model_type}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeSection === 'billing' && billing && (
            <div className={styles.section}>
              <h2>Billing & Plans</h2>
              <div className={styles.plansGrid}>
                <div className={`${styles.planCard} ${billing.plan_type === 'free' ? styles.activePlan : ''}`}>
                  <h3>Free</h3>
                  <div className={styles.planPrice}>$0</div>
                  <ul>
                    <li>1 AI Model</li>
                    <li>Basic Support</li>
                  </ul>
                  {billing.plan_type === 'free' && <div className={styles.currentPlan}>Current Plan</div>}
                </div>
                <div className={`${styles.planCard} ${billing.plan_type === 'pro' ? styles.activePlan : ''}`}>
                  <h3>Pro</h3>
                  <div className={styles.planPrice}>$50</div>
                  <ul>
                    <li>10 AI Models</li>
                    <li>Priority Support</li>
                  </ul>
                  {billing.plan_type === 'pro' && <div className={styles.currentPlan}>Current Plan</div>}
                </div>
                <div className={`${styles.planCard} ${billing.plan_type === 'enterprise' ? styles.activePlan : ''}`}>
                  <h3>Enterprise</h3>
                  <div className={styles.planPrice}>$450</div>
                  <ul>
                    <li>30 AI Models</li>
                    <li>API Access</li>
                    <li>24/7 Support</li>
                  </ul>
                  {billing.plan_type === 'enterprise' && <div className={styles.currentPlan}>Current Plan</div>}
                </div>
              </div>
              <div className={styles.usageStats}>
                <p>Models Created: {billing.models_created} / {billing.models_limit}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {showCreateModal && (
        <CreateModelModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            loadModels();
            loadBilling();
          }}
        />
      )}
    </div>
  );
}

// Create Model Modal Component
function CreateModelModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const { user } = useAuth();
  const [prompt, setPrompt] = useState('');
  const [trainingMode, setTrainingMode] = useState<'from_scratch' | 'fine_tune'>('from_scratch');
  const [uploadedDataset, setUploadedDataset] = useState<File | null>(null);
  const [uploadedModel, setUploadedModel] = useState<File | null>(null);
  const [extraInstructions, setExtraInstructions] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || !user) return;
    if (trainingMode === 'from_scratch' && !uploadedDataset) {
      alert('Dataset is required for "from scratch" training');
      return;
    }

    setIsCreating(true);
    // TODO: Implement model creation logic
    setTimeout(() => {
      setIsCreating(false);
      onSuccess();
    }, 2000);
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2>Create AI Model</h2>
        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label>Describe your AI model</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Create a sentiment analysis model using BERT"
              rows={3}
            />
          </div>

          <div className={styles.formGroup}>
            <label>Training Mode</label>
            <div className={styles.toggleGroup}>
              <button
                type="button"
                className={trainingMode === 'from_scratch' ? styles.active : ''}
                onClick={() => setTrainingMode('from_scratch')}
              >
                From Scratch
              </button>
              <button
                type="button"
                className={trainingMode === 'fine_tune' ? styles.active : ''}
                onClick={() => setTrainingMode('fine_tune')}
              >
                Fine Tune
              </button>
            </div>
          </div>

          <div className={styles.formGroup}>
            <label>
              Use Your Own Data {trainingMode === 'from_scratch' && <span className={styles.required}>*</span>}
            </label>
            <input
              type="file"
              accept=".csv,.json,.txt,.parquet"
              onChange={(e) => setUploadedDataset(e.target.files?.[0] || null)}
            />
          </div>

          <div className={styles.formGroup}>
            <label>Upload Your Own AI Model</label>
            <input
              type="file"
              accept=".pth,.h5,.pb,.onnx,.safetensors"
              onChange={(e) => setUploadedModel(e.target.files?.[0] || null)}
            />
          </div>

          <div className={styles.formGroup}>
            <label>Extra Instructions (Optional)</label>
            <textarea
              value={extraInstructions}
              onChange={(e) => setExtraInstructions(e.target.value)}
              placeholder="Any additional requirements..."
              rows={2}
            />
          </div>

          <div className={styles.modalActions}>
            <button type="button" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={isCreating}>
              {isCreating ? 'Creating...' : 'Create Model'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

