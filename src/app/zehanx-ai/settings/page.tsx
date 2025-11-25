'use client';

import React, { useState } from 'react';
import { Save, AlertCircle, CheckCircle } from 'lucide-react';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    defaultModelType: 'transformer',
    defaultDatasetSource: 'firecrawl',
    defaultEpochs: 10,
    defaultBatchSize: 32,
    defaultLearningRate: 0.001,
    autoDownloadModels: true,
    enableNotifications: true,
  });

  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem('zehanx-ai-settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="p-6 space-y-6">
      <div className="space-y-2 mb-8">
        <h2 className="text-3xl font-bold text-white">Settings</h2>
        <p className="text-slate-400">Configure your Zehanx AI preferences</p>
      </div>

      <div className="space-y-6">
        {/* Default Model Settings */}
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
          <h3 className="text-xl font-semibold text-white">Default Model Settings</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Default Model Type</label>
              <select
                value={settings.defaultModelType}
                onChange={(e) => setSettings({ ...settings, defaultModelType: e.target.value })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="transformer">Transformer</option>
                <option value="lstm">LSTM</option>
                <option value="cnn">CNN</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Default Dataset Source</label>
              <select
                value={settings.defaultDatasetSource}
                onChange={(e) => setSettings({ ...settings, defaultDatasetSource: e.target.value })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="firecrawl">Firecrawl</option>
                <option value="github">GitHub</option>
                <option value="huggingface">Hugging Face</option>
                <option value="kaggle">Kaggle</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Default Epochs</label>
              <input
                type="number"
                min="1"
                max="100"
                value={settings.defaultEpochs}
                onChange={(e) => setSettings({ ...settings, defaultEpochs: parseInt(e.target.value) })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Default Batch Size</label>
              <input
                type="number"
                min="1"
                max="256"
                value={settings.defaultBatchSize}
                onChange={(e) => setSettings({ ...settings, defaultBatchSize: parseInt(e.target.value) })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-slate-300 mb-2">Default Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                min="0.00001"
                max="0.1"
                value={settings.defaultLearningRate}
                onChange={(e) => setSettings({ ...settings, defaultLearningRate: parseFloat(e.target.value) })}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Preferences */}
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
          <h3 className="text-xl font-semibold text-white">Preferences</h3>

          <div className="space-y-3">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.autoDownloadModels}
                onChange={(e) => setSettings({ ...settings, autoDownloadModels: e.target.checked })}
                className="w-4 h-4 rounded bg-slate-700 border border-slate-600 cursor-pointer"
              />
              <span className="text-slate-300">Auto-download trained models</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.enableNotifications}
                onChange={(e) => setSettings({ ...settings, enableNotifications: e.target.checked })}
                className="w-4 h-4 rounded bg-slate-700 border border-slate-600 cursor-pointer"
              />
              <span className="text-slate-300">Enable notifications</span>
            </label>
          </div>
        </div>

        {/* Information */}
        <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-lg p-6 space-y-3">
          <div className="flex items-start gap-3">
            <AlertCircle size={20} className="text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-white mb-1">About Zehanx AI</h4>
              <p className="text-slate-300 text-sm">
                Zehanx AI is a complete AI model generation platform integrated with your ADVANCED_WEBSITE_DESIGN project.
                It allows you to create, train, and deploy custom AI models with multiple architectures and dataset sources.
              </p>
            </div>
          </div>
        </div>

        {/* System Information */}
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 space-y-4">
          <h3 className="text-xl font-semibold text-white">System Information</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { label: 'Platform', value: 'Zehanx AI v1.0' },
              { label: 'Status', value: 'Production Ready' },
              { label: 'Framework', value: 'Next.js 15 + React 19' },
              { label: 'ML Framework', value: 'PyTorch' },
              { label: 'Data Fetching', value: 'Firecrawl API' },
              { label: 'Sandbox', value: 'E2B' },
            ].map((info, idx) => (
              <div key={idx} className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-slate-400 text-sm mb-1">{info.label}</p>
                <p className="text-white font-medium">{info.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Save Button */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleSave}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all flex items-center gap-2"
          >
            <Save size={20} />
            Save Settings
          </button>

          {saved && (
            <div className="flex items-center gap-2 text-green-400">
              <CheckCircle size={20} />
              <span>Settings saved successfully!</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
