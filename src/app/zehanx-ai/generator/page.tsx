'use client';

import React, { useState } from 'react';
import { Loader2, Github, BookOpen, Zap } from 'lucide-react';
import { toast } from 'sonner';

interface TrainingStats {
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: string;
}

interface ModelConfig {
  name: string;
  description: string;
  modelType: 'transformer' | 'lstm' | 'cnn' | 'custom';
  datasetSource: 'firecrawl' | 'huggingface' | 'kaggle' | 'github';
  githubRepo?: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

export default function ModelGeneratorPage() {
  const [step, setStep] = useState<'config' | 'training' | 'complete'>('config');
  const [isLoading, setIsLoading] = useState(false);
  const [trainingStats, setTrainingStats] = useState<TrainingStats[]>([]);
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    name: 'My AI Model',
    description: 'Custom AI model',
    modelType: 'transformer',
    datasetSource: 'firecrawl',
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
  });
  const [githubRepoUrl, setGithubRepoUrl] = useState('');

  const handleStartTraining = async () => {
    if (!modelConfig.name.trim()) {
      toast.error('Please enter a model name');
      return;
    }

    if (modelConfig.datasetSource === 'github' && !githubRepoUrl.trim()) {
      toast.error('Please enter a GitHub repository URL');
      return;
    }

    setIsLoading(true);
    setStep('training');
    setTrainingStats([]);

    try {
      const response = await fetch('/api/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...modelConfig,
          githubRepo: githubRepoUrl,
        }),
      });

      if (!response.ok) throw new Error('Failed to start training');

      if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(line => line.trim());

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'stats') {
                  setTrainingStats(prev => [...prev, data.stats]);
                } else if (data.type === 'complete') {
                  setStep('complete');
                  toast.success('Model training completed!');
                } else if (data.type === 'error') {
                  toast.error(data.message);
                  setStep('config');
                }
              } catch (e) {
                console.error('Failed to parse stats:', e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Training error:', error);
      toast.error('Failed to start training');
      setStep('config');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {step === 'config' && (
        <>
          <div className="space-y-2 mb-8">
            <h2 className="text-3xl font-bold text-white">Create AI Model</h2>
            <p className="text-slate-400">Configure and train your custom AI model</p>
          </div>

          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-6">
            {/* Model Name */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Model Name</label>
              <input
                type="text"
                value={modelConfig.name}
                onChange={(e) => setModelConfig({ ...modelConfig, name: e.target.value })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="Enter model name"
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Description</label>
              <textarea
                value={modelConfig.description}
                onChange={(e) => setModelConfig({ ...modelConfig, description: e.target.value })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none h-24"
                placeholder="Describe your model"
              />
            </div>

            {/* Model Type */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Model Type</label>
              <select
                value={modelConfig.modelType}
                onChange={(e) =>
                  setModelConfig({
                    ...modelConfig,
                    modelType: e.target.value as ModelConfig['modelType'],
                  })
                }
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="transformer">Transformer - Best for NLP</option>
                <option value="lstm">LSTM - Best for Time Series</option>
                <option value="cnn">CNN - Best for Images</option>
                <option value="custom">Custom - General Purpose</option>
              </select>
            </div>

            {/* Dataset Source */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-3">Dataset Source</label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: 'firecrawl', label: 'Firecrawl', icon: BookOpen },
                  { id: 'huggingface', label: 'Hugging Face', icon: Zap },
                  { id: 'kaggle', label: 'Kaggle', icon: Zap },
                  { id: 'github', label: 'GitHub Repo', icon: Github },
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() =>
                      setModelConfig({
                        ...modelConfig,
                        datasetSource: id as ModelConfig['datasetSource'],
                      })
                    }
                    className={`p-3 rounded-lg border-2 transition-all flex items-center gap-2 ${
                      modelConfig.datasetSource === id
                        ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                        : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
                    }`}
                  >
                    <Icon size={18} />
                    <span className="text-sm font-medium">{label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* GitHub Repo URL */}
            {modelConfig.datasetSource === 'github' && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">GitHub Repository URL</label>
                <input
                  type="text"
                  value={githubRepoUrl}
                  onChange={(e) => setGithubRepoUrl(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  placeholder="https://github.com/username/repo"
                />
              </div>
            )}

            {/* Training Parameters */}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={modelConfig.epochs}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      epochs: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Batch Size</label>
                <input
                  type="number"
                  min="1"
                  max="256"
                  value={modelConfig.batchSize}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      batchSize: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  min="0.00001"
                  max="0.1"
                  value={modelConfig.learningRate}
                  onChange={(e) =>
                    setModelConfig({
                      ...modelConfig,
                      learningRate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Start Button */}
            <button
              onClick={handleStartTraining}
              disabled={isLoading}
              className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-slate-600 disabled:to-slate-700 text-white font-semibold rounded-lg transition-all flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  Starting Training...
                </>
              ) : (
                'Start Training'
              )}
            </button>
          </div>
        </>
      )}

      {step === 'training' && (
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Loader2 size={24} className="animate-spin text-blue-400" />
            Training in Progress
          </h2>

          {trainingStats.length > 0 && (
            <div className="space-y-4">
              {/* Stats Summary */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-slate-400 text-sm mb-1">Current Epoch</p>
                  <p className="text-2xl font-bold text-white">
                    {trainingStats[trainingStats.length - 1].epoch}
                  </p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-slate-400 text-sm mb-1">Loss</p>
                  <p className="text-2xl font-bold text-red-400">
                    {trainingStats[trainingStats.length - 1].loss.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-slate-400 text-sm mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-green-400">
                    {(trainingStats[trainingStats.length - 1].accuracy * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {/* Training Log */}
              <div className="bg-slate-900/50 rounded-lg p-4 max-h-96 overflow-y-auto">
                <div className="space-y-2 font-mono text-sm">
                  {trainingStats.map((stat, idx) => (
                    <div key={idx} className="text-slate-300">
                      <span className="text-blue-400">Epoch {stat.epoch}</span>
                      <span className="text-slate-500"> | </span>
                      <span className="text-red-400">Loss: {stat.loss.toFixed(4)}</span>
                      <span className="text-slate-500"> | </span>
                      <span className="text-green-400">Acc: {(stat.accuracy * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {step === 'complete' && (
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 text-center space-y-6">
          <div>
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500/20 rounded-full mb-4">
              <span className="text-3xl">✓</span>
            </div>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Training Complete!</h2>
            <p className="text-slate-400">Your model has been successfully trained and saved.</p>
          </div>

          {trainingStats.length > 0 && (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-1">Total Epochs</p>
                <p className="text-2xl font-bold text-white">
                  {trainingStats[trainingStats.length - 1].epoch}
                </p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-1">Final Loss</p>
                <p className="text-2xl font-bold text-red-400">
                  {trainingStats[trainingStats.length - 1].loss.toFixed(4)}
                </p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-slate-400 text-sm mb-1">Final Accuracy</p>
                <p className="text-2xl font-bold text-green-400">
                  {(trainingStats[trainingStats.length - 1].accuracy * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          )}

          <button
            onClick={() => {
              setStep('config');
              setTrainingStats([]);
            }}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all"
          >
            Train Another Model
          </button>
        </div>
      )}
    </div>
  );
}
