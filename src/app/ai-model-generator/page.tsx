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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            🤖 AI Model Generator
          </h1>
          <p className="text-lg text-gray-400">
            Describe your AI model, and we'll generate, train, and deploy it for you
          </p>
        </div>

        {/* Main Content */}
        {!deploymentResult ? (
          <div className="space-y-8">
            {/* Model Selector */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg p-6">
              <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
            </div>

            {/* Prompt Input */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg p-6">
              <PromptInput onSubmit={handleSubmit} isLoading={isLoading} />
            </div>

            {/* Progress Display */}
            {isLoading && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg p-6">
                <ProgressDisplay steps={steps} currentStep={steps.findIndex((s: Step) => s.status === 'in-progress')} error={error || undefined} />
              </div>
            )}

            {/* Error Display */}
            {error && !isLoading && (
              <div className="bg-red-500/10 backdrop-blur border border-red-500 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-red-400 mb-2">❌ Error</h3>
                <p className="text-red-300">{error}</p>
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
                  className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-all"
                >
                  Try Again
                </button>
              </div>
            )}
          </div>
        ) : (
          /* Deployment Result */
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg p-6">
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
  );
}
