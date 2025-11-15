'use client';

import React, { useState } from 'react';
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
  const [selectedModel, setSelectedModel] = useState('mixtral-8x7b-32768');
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

        .ai-generator-wrapper {
          width: 100%;
          min-height: 100vh;
          background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #111111 100%);
          padding: 2rem;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
        }

        .ai-generator-content {
          width: 100%;
          max-width: 56rem;
        }

        .ai-generator-header {
          margin-bottom: 3rem;
          text-align: center;
        }

        .ai-generator-title {
          font-size: 3rem;
          font-weight: 700;
          color: #ffffff;
          margin-bottom: 1rem;
          line-height: 1.2;
        }

        .ai-generator-subtitle {
          font-size: 1.125rem;
          color: #9ca3af;
          line-height: 1.6;
        }

        .ai-generator-sections {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .ai-generator-card {
          background: rgba(31, 41, 55, 0.5);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(55, 65, 81, 0.5);
          border-radius: 0.5rem;
          padding: 1.5rem;
          transition: all 0.3s ease;
        }

        .ai-generator-card:hover {
          border-color: rgba(55, 65, 81, 0.8);
          background: rgba(31, 41, 55, 0.7);
        }

        .ai-generator-error-card {
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.5);
        }

        .ai-generator-error-title {
          font-size: 1.125rem;
          font-weight: 600;
          color: #f87171;
          margin-bottom: 0.5rem;
        }

        .ai-generator-error-message {
          color: #fca5a5;
          margin-bottom: 1rem;
          line-height: 1.6;
        }

        .ai-generator-button {
          margin-top: 1rem;
          padding: 0.75rem 1rem;
          background: #dc2626;
          color: white;
          border: none;
          border-radius: 0.375rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .ai-generator-button:hover {
          background: #b91c1c;
          transform: translateY(-2px);
        }

        .ai-generator-button:active {
          transform: translateY(0);
        }

        @media (max-width: 768px) {
          .ai-generator-wrapper {
            padding: 1rem;
          }

          .ai-generator-title {
            font-size: 2rem;
          }

          .ai-generator-subtitle {
            font-size: 1rem;
          }

          .ai-generator-card {
            padding: 1rem;
          }
        }
      `}</style>

      <div className="ai-generator-wrapper">
        <div className="ai-generator-content">
          {/* Header */}
          <div className="ai-generator-header">
            <h1 className="ai-generator-title">🤖 AI Model Generator</h1>
            <p className="ai-generator-subtitle">
              Describe your AI model, and we'll generate, train, and deploy it for you
            </p>
          </div>

          {/* Main Content */}
          {!deploymentResult ? (
            <div className="ai-generator-sections">
              {/* Model Selector */}
              <div className="ai-generator-card">
                <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
              </div>

              {/* Prompt Input */}
              <div className="ai-generator-card">
                <PromptInput onSubmit={handleSubmit} isLoading={isLoading} />
              </div>

              {/* Progress Display */}
              {isLoading && (
                <div className="ai-generator-card">
                  <ProgressDisplay steps={steps} currentStep={steps.findIndex((s: Step) => s.status === 'in-progress')} error={error || undefined} />
                </div>
              )}

              {/* Error Display */}
              {error && !isLoading && (
                <div className="ai-generator-card ai-generator-error-card">
                  <h3 className="ai-generator-error-title">❌ Error</h3>
                  <p className="ai-generator-error-message">{error}</p>
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
                    className="ai-generator-button"
                  >
                    Try Again
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* Deployment Result */
            <div className="ai-generator-card">
              <DeploymentResult
                deploymentUrl={deploymentResult.deploymentUrl}
                sandboxId={deploymentResult.sandboxId}
                modelType={deploymentResult.modelType}
                endpoints={deploymentResult.endpoints}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
