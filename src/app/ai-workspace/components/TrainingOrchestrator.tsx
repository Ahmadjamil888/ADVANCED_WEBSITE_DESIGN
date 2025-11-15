'use client';

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';

interface TrainingPlan {
  task: string;
  recommendedModels: Array<{
    name: string;
    framework: string;
    pretrained: string;
    reason: string;
  }>;
  dataset: {
    name: string;
    source: string;
    url: string;
    size: string;
  };
  estimatedTime: string;
  dependencies: string[];
}

interface TrainingOrchestratorProps {
  modelId: string;
  onTrainingStart: (trainingJobId: string) => void;
}

export function TrainingOrchestrator({
  modelId,
  onTrainingStart,
}: TrainingOrchestratorProps) {
  const { user } = useAuth();
  const [step, setStep] = useState<'prompt' | 'plan' | 'model-select' | 'training'>('prompt');
  const [prompt, setPrompt] = useState('');
  const [plan, setPlan] = useState<TrainingPlan | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [trainingJobId, setTrainingJobId] = useState<string>('');

  const handleGeneratePlan = async () => {
    if (!prompt.trim()) {
      setError('Please describe your model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/training/orchestrator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'generate-plan',
          task: prompt,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate plan');
      }

      setPlan(data.plan);
      setStep('model-select');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    if (!user) {
      setError('Please sign in');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/training/orchestrator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start-training',
          modelId,
          userId: user.id,
          selectedModel,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to start training');
      }

      setTrainingJobId(data.trainingJobId);
      setStep('training');
      onTrainingStart(data.trainingJobId);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      {/* Step 1: Prompt */}
      {step === 'prompt' && (
        <div>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
            📝 Describe Your Model
          </h2>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Create a sentiment analysis model that can classify text as positive, negative, or neutral"
            style={{
              width: '100%',
              height: '120px',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #ddd',
              fontSize: '14px',
              fontFamily: 'monospace',
            }}
          />

          {error && (
            <div
              style={{
                marginTop: '12px',
                padding: '12px',
                backgroundColor: '#fee',
                color: '#c33',
                borderRadius: '4px',
              }}
            >
              ❌ {error}
            </div>
          )}

          <button
            onClick={handleGeneratePlan}
            disabled={loading}
            style={{
              marginTop: '20px',
              padding: '12px 24px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
              fontSize: '16px',
              fontWeight: 'bold',
            }}
          >
            {loading ? '⏳ Generating Plan...' : '🚀 Generate Training Plan'}
          </button>
        </div>
      )}

      {/* Step 2: Model Selection */}
      {step === 'model-select' && plan && (
        <div>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
            🤖 Select Model
          </h2>

          <div style={{ marginBottom: '20px' }}>
            <p style={{ color: '#666', marginBottom: '10px' }}>
              <strong>Task:</strong> {plan.task}
            </p>
            <p style={{ color: '#666', marginBottom: '10px' }}>
              <strong>Dataset:</strong> {plan.dataset.name} ({plan.dataset.size})
            </p>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              <strong>Estimated Time:</strong> {plan.estimatedTime}
            </p>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '12px' }}>
              Recommended Models:
            </h3>

            <div style={{ display: 'grid', gap: '12px' }}>
              {plan.recommendedModels.map((model, idx) => (
                <label
                  key={idx}
                  style={{
                    padding: '16px',
                    border: selectedModel === model.pretrained ? '2px solid #007bff' : '1px solid #ddd',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    backgroundColor:
                      selectedModel === model.pretrained ? '#f0f7ff' : 'white',
                  }}
                >
                  <input
                    type="radio"
                    name="model"
                    value={model.pretrained}
                    checked={selectedModel === model.pretrained}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    style={{ marginRight: '12px' }}
                  />
                  <div>
                    <strong>{model.name}</strong>
                    <p style={{ color: '#666', fontSize: '14px', margin: '4px 0' }}>
                      {model.reason}
                    </p>
                    <p style={{ color: '#999', fontSize: '12px', margin: '4px 0' }}>
                      Framework: {model.framework}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {error && (
            <div
              style={{
                marginTop: '12px',
                padding: '12px',
                backgroundColor: '#fee',
                color: '#c33',
                borderRadius: '4px',
              }}
            >
              ❌ {error}
            </div>
          )}

          <div style={{ marginTop: '20px', display: 'flex', gap: '12px' }}>
            <button
              onClick={() => setStep('prompt')}
              style={{
                padding: '12px 24px',
                backgroundColor: '#f0f0f0',
                color: '#333',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '16px',
              }}
            >
              ← Back
            </button>

            <button
              onClick={handleStartTraining}
              disabled={loading || !selectedModel}
              style={{
                padding: '12px 24px',
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading || !selectedModel ? 'not-allowed' : 'pointer',
                opacity: loading || !selectedModel ? 0.6 : 1,
                fontSize: '16px',
                fontWeight: 'bold',
              }}
            >
              {loading ? '⏳ Starting Training...' : '🏋️ Start Training'}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Training in Progress */}
      {step === 'training' && (
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
            🏋️ Training in Progress
          </h2>

          <div
            style={{
              padding: '20px',
              backgroundColor: '#f0f7ff',
              borderRadius: '8px',
              marginBottom: '20px',
            }}
          >
            <p style={{ fontSize: '16px', color: '#0066cc' }}>
              Training Job ID: <code>{trainingJobId}</code>
            </p>
            <p style={{ color: '#666', marginTop: '10px' }}>
              Check the training tab for real-time progress...
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
