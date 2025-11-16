'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import ModelSelector from '@/components/AIModelGenerator/ModelSelector';
import PromptInput from '@/components/AIModelGenerator/PromptInput';
import ProgressDisplay from '@/components/AIModelGenerator/ProgressDisplay';
import DeploymentResult from '@/components/AIModelGenerator/DeploymentResult';

interface Step {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  details?: string;
}

export default function AIModelGeneratorPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'generator' | 'models' | 'billing'>('generator');
  const [selectedModel, setSelectedModel] = useState('openai/gpt-oss-120b');
  const [isLoading, setIsLoading] = useState(false);
  const [steps, setSteps] = useState<Step[]>([
    { name: 'Code Generation', status: 'pending' },
    { name: 'Sandbox Creation', status: 'pending' },
    { name: 'Model Training', status: 'pending' },
    { name: 'E2B Deployment', status: 'pending' },
  ]);
  const [error, setError] = useState<string | null>(null);
  const [deploymentResult, setDeploymentResult] = useState<any>(null);

  const updateStep = (index: number, status: 'pending' | 'in-progress' | 'completed' | 'error', details?: string) => {
    setSteps((prev: Step[]) => {
      const newSteps = [...prev];
      newSteps[index] = { ...newSteps[index], status, details };
      return newSteps;
    });
  };

  const handleSubmit = async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    setDeploymentResult(null);
    setSteps([
      { name: 'Code Generation', status: 'in-progress' },
      { name: 'Sandbox Creation', status: 'pending' },
      { name: 'Model Training', status: 'pending' },
      { name: 'E2B Deployment', status: 'pending' },
    ]);

    try {
      console.log('Starting orchestration with prompt:', prompt);

      const response = await fetch('/api/ai/orchestrate-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model: selectedModel }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate and train model');
      }

      const data = await response.json();
      console.log('Orchestration response:', data);

      // Update steps based on response
      updateStep(0, 'completed', `Generated code with ${data.steps.codeGeneration.modelType} model`);
      updateStep(1, 'completed', `Sandbox: ${data.sandboxId.slice(0, 8)}...`);
      updateStep(2, 'completed', 'Model trained successfully');
      updateStep(3, 'completed', 'Deployed to E2B');

      setDeploymentResult({
        deploymentUrl: data.deploymentUrl,
        sandboxId: data.sandboxId,
        modelType: data.modelType,
        endpoints: data.steps.deployment.endpoints,
      });
    } catch (err) {
      console.error('Error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);

      // Mark current step as error
      const currentStep = steps.findIndex((s: Step) => s.status === 'in-progress');
      if (currentStep !== -1) {
        updateStep(currentStep, 'error', errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await fetch('/api/auth/signout', { method: 'POST' });
      router.push('/login');
    } catch (err) {
      console.error('Sign out error:', err);
    }
  };

  return (
    <div>
      <style>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body, html {
          width: 100%;
          height: 100%;
        }

        .dashboard-wrapper {
          width: 100%;
          min-height: 100vh;
          background: #000000;
          display: flex;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }

        .dashboard-sidebar {
          width: 250px;
          background: #0a0a0a;
          border-right: 1px solid #222222;
          padding: 1.5rem 0;
          display: flex;
          flex-direction: column;
        }

        .dashboard-logo {
          padding: 0 1.5rem 2rem;
          font-size: 1.25rem;
          font-weight: 700;
          color: #ffffff;
          border-bottom: 1px solid #222222;
          margin-bottom: 1.5rem;
        }

        .dashboard-nav {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          padding: 0 1rem;
        }

        .dashboard-nav-item {
          padding: 0.75rem 1rem;
          background: transparent;
          border: 1px solid transparent;
          color: #999999;
          cursor: pointer;
          font-size: 0.95rem;
          transition: all 0.2s ease;
          text-align: left;
          font-family: inherit;
        }

        .dashboard-nav-item:hover {
          background: #111111;
          color: #ffffff;
          border-color: #333333;
        }

        .dashboard-nav-item.active {
          background: #111111;
          color: #ffffff;
          border-color: #444444;
        }

        .dashboard-footer {
          padding: 1rem;
          border-top: 1px solid #222222;
        }

        .dashboard-signout-btn {
          width: 100%;
          padding: 0.75rem 1rem;
          background: #1a1a1a;
          border: 1px solid #333333;
          color: #ffffff;
          cursor: pointer;
          font-size: 0.9rem;
          transition: all 0.2s ease;
          font-family: inherit;
        }

        .dashboard-signout-btn:hover {
          background: #222222;
          border-color: #444444;
        }

        .dashboard-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          background: #000000;
        }

        .dashboard-header {
          padding: 1.5rem 2rem;
          border-bottom: 1px solid #222222;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .dashboard-title {
          font-size: 1.5rem;
          font-weight: 700;
          color: #ffffff;
        }

        .dashboard-content {
          flex: 1;
          padding: 0;
          overflow-y: auto;
          width: 100%;
          height: 100%;
        }

        .dashboard-tabs {
          display: flex;
          gap: 0;
          border-bottom: 1px solid #222222;
          margin-bottom: 2rem;
        }

        .dashboard-tab {
          padding: 1rem 1.5rem;
          background: transparent;
          border: none;
          border-bottom: 2px solid transparent;
          color: #666666;
          cursor: pointer;
          font-size: 0.95rem;
          transition: all 0.2s ease;
          font-family: inherit;
          font-weight: 500;
        }

        .dashboard-tab:hover {
          color: #ffffff;
        }

        .dashboard-tab.active {
          color: #ffffff;
          border-bottom-color: #ffffff;
        }

        .dashboard-card {
          background: #0a0a0a;
          border: 1px solid #222222;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .dashboard-card-title {
          font-size: 1rem;
          font-weight: 600;
          color: #ffffff;
          margin-bottom: 1rem;
        }

        .dashboard-error-card {
          background: #1a0a0a;
          border-color: #4a1a1a;
        }

        .dashboard-error-title {
          color: #ff6b6b;
        }

        .dashboard-error-message {
          color: #ff9999;
          margin-bottom: 1rem;
          font-size: 0.9rem;
        }

        .dashboard-button {
          padding: 0.75rem 1.5rem;
          background: #ffffff;
          border: 1px solid #ffffff;
          color: #000000;
          cursor: pointer;
          font-weight: 600;
          font-size: 0.9rem;
          transition: all 0.2s ease;
          font-family: inherit;
        }

        .dashboard-button:hover {
          background: #f0f0f0;
          border-color: #f0f0f0;
        }

        .dashboard-button.secondary {
          background: transparent;
          border-color: #333333;
          color: #ffffff;
        }

        .dashboard-button.secondary:hover {
          background: #111111;
          border-color: #555555;
        }

        .billing-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .billing-item {
          background: #0a0a0a;
          border: 1px solid #222222;
          padding: 1.5rem;
        }

        .billing-label {
          font-size: 0.85rem;
          color: #666666;
          margin-bottom: 0.5rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .billing-value {
          font-size: 2rem;
          font-weight: 700;
          color: #ffffff;
        }

        .billing-unit {
          font-size: 0.85rem;
          color: #999999;
          margin-top: 0.5rem;
        }

        @media (max-width: 768px) {
          .dashboard-wrapper {
            flex-direction: column;
          }

          .dashboard-sidebar {
            width: 100%;
            border-right: none;
            border-bottom: 1px solid #222222;
            padding: 1rem;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
          }

          .dashboard-logo {
            padding: 0;
            border: none;
            margin: 0;
          }

          .dashboard-nav {
            flex-direction: row;
            padding: 0;
            gap: 0;
          }

          .dashboard-nav-item {
            padding: 0.5rem 1rem;
            border: none;
          }

          .dashboard-footer {
            display: none;
          }

          .dashboard-content {
            padding: 1rem;
          }

          .billing-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>

      <div className="dashboard-wrapper">
        {/* Sidebar */}
        <div className="dashboard-sidebar">
          <div className="dashboard-logo">AI Studio</div>
          <div className="dashboard-nav">
            <button
              className={`dashboard-nav-item ${activeTab === 'generator' ? 'active' : ''}`}
              onClick={() => setActiveTab('generator')}
            >
              Generator
            </button>
            <button
              className={`dashboard-nav-item ${activeTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveTab('models')}
            >
              My Models
            </button>
            <button
              className={`dashboard-nav-item ${activeTab === 'billing' ? 'active' : ''}`}
              onClick={() => setActiveTab('billing')}
            >
              Billing
            </button>
          </div>
          <div className="dashboard-footer">
            <button className="dashboard-signout-btn" onClick={handleSignOut}>
              Sign Out
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="dashboard-main">
          <div className="dashboard-header">
            <h1 className="dashboard-title">
              {activeTab === 'generator' && 'AI Model Generator'}
              {activeTab === 'models' && 'My Models'}
              {activeTab === 'billing' && 'Billing & Usage'}
            </h1>
          </div>

          <div className="dashboard-content">
            {/* Generator Tab */}
            {activeTab === 'generator' && (
              <div>
                {!deploymentResult ? (
                  <div>
                    <div className="dashboard-card">
                      <div className="dashboard-card-title">Select Model</div>
                      <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
                    </div>

                    <div className="dashboard-card">
                      <div className="dashboard-card-title">Create Model</div>
                      <PromptInput onSubmit={handleSubmit} isLoading={isLoading} />
                    </div>

                    {isLoading && (
                      <div className="dashboard-card">
                        <div className="dashboard-card-title">Progress</div>
                        <ProgressDisplay steps={steps} currentStep={steps.findIndex((s: Step) => s.status === 'in-progress')} error={error || undefined} />
                      </div>
                    )}

                    {error && !isLoading && (
                      <div className="dashboard-card dashboard-error-card">
                        <div className="dashboard-error-title">Error</div>
                        <p className="dashboard-error-message">{error}</p>
                        <button
                          onClick={() => {
                            setError(null);
                            setSteps([
                              { name: 'Code Generation', status: 'pending' },
                              { name: 'Sandbox Creation', status: 'pending' },
                              { name: 'Model Training', status: 'pending' },
                              { name: 'E2B Deployment', status: 'pending' },
                            ]);
                          }}
                          className="dashboard-button"
                        >
                          Try Again
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="dashboard-card">
                    <div className="dashboard-card-title">Deployment Result</div>
                    <DeploymentResult
                      deploymentUrl={deploymentResult.deploymentUrl}
                      sandboxId={deploymentResult.sandboxId}
                      modelType={deploymentResult.modelType}
                      endpoints={deploymentResult.endpoints}
                    />
                  </div>
                )}
              </div>
            )}

            {/* My Models Tab */}
            {activeTab === 'models' && (
              <div>
                <div className="dashboard-card">
                  <div className="dashboard-card-title">Your Models</div>
                  <p style={{ color: '#666666', fontSize: '0.9rem' }}>
                    No models created yet. Start by generating your first model.
                  </p>
                </div>
              </div>
            )}

            {/* Billing Tab */}
            {activeTab === 'billing' && (
              <div>
                <div className="billing-grid">
                  <div className="billing-item">
                    <div className="billing-label">API Calls</div>
                    <div className="billing-value">0</div>
                    <div className="billing-unit">this month</div>
                  </div>
                  <div className="billing-item">
                    <div className="billing-label">Compute Hours</div>
                    <div className="billing-value">0</div>
                    <div className="billing-unit">used</div>
                  </div>
                  <div className="billing-item">
                    <div className="billing-label">Current Plan</div>
                    <div className="billing-value">Free</div>
                    <div className="billing-unit">tier</div>
                  </div>
                </div>

                <div className="dashboard-card">
                  <div className="dashboard-card-title">Usage Details</div>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #222222' }}>
                        <th style={{ textAlign: 'left', padding: '0.75rem', color: '#999999', fontSize: '0.85rem', fontWeight: 600 }}>Service</th>
                        <th style={{ textAlign: 'left', padding: '0.75rem', color: '#999999', fontSize: '0.85rem', fontWeight: 600 }}>Usage</th>
                        <th style={{ textAlign: 'left', padding: '0.75rem', color: '#999999', fontSize: '0.85rem', fontWeight: 600 }}>Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{ borderBottom: '1px solid #1a1a1a' }}>
                        <td style={{ padding: '0.75rem', color: '#ffffff' }}>Code Generation</td>
                        <td style={{ padding: '0.75rem', color: '#999999' }}>0 calls</td>
                        <td style={{ padding: '0.75rem', color: '#999999' }}>$0.00</td>
                      </tr>
                      <tr style={{ borderBottom: '1px solid #1a1a1a' }}>
                        <td style={{ padding: '0.75rem', color: '#ffffff' }}>Model Training</td>
                        <td style={{ padding: '0.75rem', color: '#999999' }}>0 hours</td>
                        <td style={{ padding: '0.75rem', color: '#999999' }}>$0.00</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.75rem', color: '#ffffff', fontWeight: 600 }}>Total</td>
                        <td style={{ padding: '0.75rem', color: '#999999' }}></td>
                        <td style={{ padding: '0.75rem', color: '#ffffff', fontWeight: 600 }}>$0.00</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
