'use client';

import React, { useState, useEffect } from 'react';
import { Download, Trash2, Eye, BarChart3 } from 'lucide-react';

interface ModelStats {
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: string;
}

interface TrainedModel {
  name: string;
  type: string;
  createdAt: string;
  finalLoss: number;
  finalAccuracy: number;
  epochs: number;
  fileSize: number;
  stats: ModelStats[];
}

export default function ModelsPage() {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadModel = async (modelName: string) => {
    try {
      const response = await fetch(`/api/download-model?name=${modelName}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${modelName}.pt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  const deleteModel = async (modelName: string) => {
    if (!confirm(`Delete model "${modelName}"?`)) return;

    try {
      const response = await fetch(`/api/delete-model?name=${modelName}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setModels(models.filter(m => m.name !== modelName));
        setSelectedModel(null);
      }
    } catch (error) {
      console.error('Failed to delete model:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="space-y-2 mb-8">
        <h2 className="text-3xl font-bold text-white">My Models</h2>
        <p className="text-slate-400">View, download, and manage your trained AI models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Models List */}
        <div className="lg:col-span-1">
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-4 space-y-3">
            <h3 className="text-lg font-semibold text-white">Trained Models</h3>

            {isLoading ? (
              <div className="text-slate-400 text-center py-8">Loading models...</div>
            ) : models.length === 0 ? (
              <div className="text-slate-400 text-sm text-center py-8">
                No trained models yet. <br /> Go to Model Generator to create one.
              </div>
            ) : (
              <div className="space-y-2">
                {models.map((model) => (
                  <button
                    key={model.name}
                    onClick={() => setSelectedModel(model)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedModel?.name === model.name
                        ? 'bg-blue-500/20 border border-blue-500 text-blue-400'
                        : 'bg-slate-700/50 border border-slate-600 text-slate-300 hover:border-slate-500'
                    }`}
                  >
                    <div className="font-medium truncate">{model.name}</div>
                    <div className="text-xs text-slate-400 mt-1">
                      Acc: {(model.finalAccuracy * 100).toFixed(1)}%
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Model Details */}
        <div className="lg:col-span-2">
          {selectedModel ? (
            <div className="space-y-4">
              {/* Header */}
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-2xl font-bold text-white">{selectedModel.name}</h3>
                    <p className="text-slate-400 text-sm mt-1">
                      Type: {selectedModel.type} • Created: {new Date(selectedModel.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => downloadModel(selectedModel.name)}
                      className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all"
                      title="Download model"
                    >
                      <Download size={18} />
                    </button>
                    <button
                      onClick={() => deleteModel(selectedModel.name)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-all"
                      title="Delete model"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-slate-400 text-xs mb-1">Final Loss</p>
                    <p className="text-xl font-bold text-red-400">
                      {selectedModel.finalLoss.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-slate-400 text-xs mb-1">Final Accuracy</p>
                    <p className="text-xl font-bold text-green-400">
                      {(selectedModel.finalAccuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-slate-400 text-xs mb-1">Epochs</p>
                    <p className="text-xl font-bold text-blue-400">{selectedModel.epochs}</p>
                  </div>
                </div>
              </div>

              {/* Training Graph */}
              {selectedModel.stats.length > 0 && (
                <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
                  <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                    <BarChart3 size={20} />
                    Training Progress
                  </h4>

                  {/* Loss Chart */}
                  <div>
                    <p className="text-sm text-slate-400 mb-2">Loss Curve</p>
                    <div className="h-32 bg-slate-900/50 rounded-lg p-2 flex items-end gap-1">
                      {selectedModel.stats.map((stat, idx) => {
                        const maxLoss = Math.max(...selectedModel.stats.map(s => s.loss));
                        const height = (stat.loss / maxLoss) * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-red-500/70 rounded-t hover:bg-red-500 transition-all"
                            style={{ height: `${height}%` }}
                            title={`Epoch ${stat.epoch}: ${stat.loss.toFixed(4)}`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Accuracy Chart */}
                  <div>
                    <p className="text-sm text-slate-400 mb-2">Accuracy Curve</p>
                    <div className="h-32 bg-slate-900/50 rounded-lg p-2 flex items-end gap-1">
                      {selectedModel.stats.map((stat, idx) => {
                        const height = stat.accuracy * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-green-500/70 rounded-t hover:bg-green-500 transition-all"
                            style={{ height: `${height}%` }}
                            title={`Epoch ${stat.epoch}: ${(stat.accuracy * 100).toFixed(2)}%`}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* Training Log */}
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-3">
                <h4 className="text-lg font-semibold text-white">Training Log</h4>
                <div className="bg-slate-900/50 rounded-lg p-4 max-h-48 overflow-y-auto font-mono text-xs space-y-1">
                  {selectedModel.stats.map((stat, idx) => (
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
          ) : (
            <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-8 text-center">
              <Eye size={48} className="mx-auto text-slate-600 mb-4" />
              <p className="text-slate-400">Select a model to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
