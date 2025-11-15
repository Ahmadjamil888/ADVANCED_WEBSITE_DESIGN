'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { getSupabaseOrThrow } from '@/lib/supabase';
import styles from './page.module.css';
import { DashboardSidebar } from './components/DashboardSidebar';
import { ModelCreationForm, ModelCreationData } from './components/ModelCreationForm';
import { ModelCard } from './components/ModelCard';
import { TrainingStats } from './components/TrainingStats';
import { ThemeToggle } from './components/ThemeToggle';
import { SignOutButton } from './components/SignOutButton';

interface AIModel {
  id: string;
  name: string;
  description?: string;
  training_status: string;
  model_type: string;
  framework: string;
  performance_metrics?: any;
  created_at: string;
  deployment_url?: string;
  model_file_path?: string;
  model_file_format?: string;
}

interface BillingInfo {
  plan_type: string;
  models_created: number;
  models_limit: number;
  has_api_access: boolean;
}

export default function AIDashboard() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [activeSection, setActiveSection] = useState('models');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [models, setModels] = useState<AIModel[]>([]);
  const [inProgressJobs, setInProgressJobs] = useState<any[]>([]);
  const [billingInfo, setBillingInfo] = useState<BillingInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!loading && user) {
      loadDashboardData();
    }
  }, [loading, user, activeSection]);

  const loadDashboardData = async () => {
    if (!user) return;
    setIsLoading(true);
    try {
      const supabase = getSupabaseOrThrow();

      if (activeSection === 'models') {
        const { data } = await supabase
          .from('ai_models')
          .select('*')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });
        setModels((data || []) as AIModel[]);
      }

      if (activeSection === 'in-progress') {
        const { data } = await supabase
          .from('training_jobs')
          .select('*')
          .eq('user_id', user.id)
          .in('job_status', ['queued', 'running'])
          .order('created_at', { ascending: false });
        setInProgressJobs(data || []);
      }

      const { data: billing } = await supabase
        .from('billing')
        .select('*')
        .eq('user_id', user.id)
        .single();
      setBillingInfo(billing as BillingInfo | null);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateModel = async (data: ModelCreationData) => {
    if (!user) return;
    setShowCreateForm(false);

    try {
      const supabase = getSupabaseOrThrow();

      // Check billing limits
      if (billingInfo && billingInfo.models_created >= billingInfo.models_limit) {
        alert(`You've reached your plan limit of ${billingInfo.models_limit} models. Please upgrade your plan.`);
        return;
      }

      // Use new orchestrated training endpoint
      console.log('[handleCreateModel] Starting orchestrated training with prompt:', data.prompt);
      
      // Show loading message
      alert('🚀 Starting AI Model Generation & Training...\n\nThis will:\n1. Generate code with Groq\n2. Create E2B sandbox\n3. Train PyTorch model\n4. Deploy REST API\n\nPlease wait...');

      // Call the new orchestration endpoint
      const response = await fetch('/api/ai/orchestrate-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: data.prompt,
          model: 'mixtral-8x7b-32768', // Default to Mixtral
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to generate and train model');
      }

      const orchestrationData = await response.json();
      console.log('[handleCreateModel] Orchestration completed:', orchestrationData);

      // Create model record in database
      const { data: model, error } = await (supabase
        .from('ai_models')
        .insert as any)({
        user_id: user.id,
        name: data.prompt.slice(0, 100),
        description: data.extraInstructions || 'Generated with AI Model Generator',
        model_type: orchestrationData.modelType,
        framework: 'pytorch',
        training_mode: 'supervised',
        training_status: 'completed',
        deployment_url: orchestrationData.deploymentUrl,
        metadata: {
          prompt: data.prompt,
          sandboxId: orchestrationData.sandboxId,
          modelType: orchestrationData.modelType,
          endpoints: orchestrationData.steps.deployment.endpoints,
          extraInstructions: data.extraInstructions,
        },
      }).select().single();

      if (error) throw error;

      console.log('[handleCreateModel] Model record created:', model.id);

      // Show success message with deployment URL
      alert(`✅ Model Successfully Generated & Deployed!\n\nDeployment URL:\n${orchestrationData.deploymentUrl}\n\nYou will be redirected to the deployment page.`);

      // Redirect to E2B deployment URL
      window.location.href = orchestrationData.deploymentUrl;
    } catch (error: any) {
      console.error('[handleCreateModel] Error:', error);
      alert(`Error: ${error.message}`);
    }
  };

  const handleDeleteModel = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return;
    try {
      const supabase = getSupabaseOrThrow();
      await supabase.from('ai_models').delete().eq('id', id);
      loadDashboardData();
    } catch (error) {
      console.error('Error deleting model:', error);
    }
  };

  const handleDownloadModel = async (id: string) => {
    try {
      const response = await fetch(`/api/models/${id}/download`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model-${id}.pth`;
        a.click();
      }
    } catch (error) {
      console.error('Error downloading model:', error);
    }
  };

  if (loading || isLoading) {
    return (
      <div className={styles.dashboard}>
        <div className={styles.loading}>Loading...</div>
      </div>
    );
  }

  if (!user) {
    router.push('/login');
    return null;
  }

  return (
    <div className={styles.dashboard}>
      <DashboardSidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      
      <div className={styles.mainContent}>
        <div className={styles.header}>
          <h1 className={styles.title}>
            {activeSection === 'models' && 'Trained Models'}
            {activeSection === 'llms' && 'LLMs'}
            {activeSection === 'datasets' && 'Datasets'}
            {activeSection === 'in-progress' && 'In Progress'}
            {activeSection === 'billing' && 'Billing'}
          </h1>
          <div className={styles.headerActions}>
            <ThemeToggle />
            {activeSection === 'models' && (
              <button className={styles.createButton} onClick={() => setShowCreateForm(true)}>
                + Create AI Model
              </button>
            )}
            <SignOutButton />
          </div>
        </div>

        <div className={styles.content}>
          {activeSection === 'models' && (
            <div className={styles.modelsGrid}>
              {models.length === 0 ? (
                <div className={styles.emptyState}>
                  <p>No models yet. Create your first AI model!</p>
                </div>
              ) : (
                models.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    onView={(id) => {
                      // Find the model to check for deployment URL
                      const selectedModel = models.find(m => m.id === id);
                      if (selectedModel?.deployment_url) {
                        // Redirect to live E2B deployment URL
                        window.location.href = selectedModel.deployment_url;
                      } else {
                        // Fallback to workspace page if no deployment URL
                        router.push(`/ai-workspace/${id}`);
                      }
                    }}
                    onDelete={handleDeleteModel}
                    onEdit={(id) => router.push(`/ai-workspace/${id}?edit=true`)}
                    onDownload={handleDownloadModel}
                  />
                ))
              )}
            </div>
          )}

          {activeSection === 'in-progress' && (
            <div className={styles.inProgressList}>
              {inProgressJobs.length === 0 ? (
                <div className={styles.emptyState}>
                  <p>No training jobs in progress.</p>
                </div>
              ) : (
                inProgressJobs.map((job) => (
                  <div key={job.id} className={styles.jobCard}>
                    <TrainingStats trainingJobId={job.id} />
                  </div>
                ))
              )}
            </div>
          )}

          {activeSection === 'billing' && billingInfo && (
            <div className={styles.billingSection}>
              <div className={styles.planCard}>
                <h2>Current Plan: {billingInfo.plan_type.toUpperCase()}</h2>
                <p>Models Created: {billingInfo.models_created} / {billingInfo.models_limit}</p>
                <p>API Access: {billingInfo.has_api_access ? 'Yes' : 'No'}</p>
              </div>
              <div className={styles.plansGrid}>
                <div className={styles.planOption}>
                  <h3>Free</h3>
                  <p className={styles.price}>$0</p>
                  <ul>
                    <li>1 AI Model</li>
                    <li>Basic Support</li>
                  </ul>
                  {billingInfo.plan_type === 'free' && <span className={styles.currentPlan}>Current</span>}
                </div>
                <div className={styles.planOption}>
                  <h3>Pro</h3>
                  <p className={styles.price}>$50</p>
                  <ul>
                    <li>10 AI Models</li>
                    <li>Priority Support</li>
                  </ul>
                  {billingInfo.plan_type === 'pro' ? (
                    <span className={styles.currentPlan}>Current</span>
                  ) : (
                    <button
                      className={styles.upgradeButton}
                      onClick={async () => {
                        const res = await fetch('/api/billing/checkout', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ planType: 'pro' }),
                        });
                        const data = await res.json();
                        if (data.url) window.location.href = data.url;
                      }}
                    >
                      Upgrade to Pro
                    </button>
                  )}
                </div>
                <div className={styles.planOption}>
                  <h3>Enterprise</h3>
                  <p className={styles.price}>$450</p>
                  <ul>
                    <li>30 AI Models</li>
                    <li>API Access</li>
                    <li>24/7 Support</li>
                  </ul>
                  {billingInfo.plan_type === 'enterprise' ? (
                    <span className={styles.currentPlan}>Current</span>
                  ) : (
                    <button
                      className={styles.upgradeButton}
                      onClick={async () => {
                        const res = await fetch('/api/billing/checkout', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ planType: 'enterprise' }),
                        });
                        const data = await res.json();
                        if (data.url) window.location.href = data.url;
                      }}
                    >
                      Upgrade to Enterprise
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {showCreateForm && (
        <ModelCreationForm
          onSubmit={handleCreateModel}
          onClose={() => setShowCreateForm(false)}
        />
      )}
    </div>
  );
}
