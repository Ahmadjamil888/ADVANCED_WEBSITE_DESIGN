'use client';

import React from 'react';
import { BookOpen, Github, Zap, Database } from 'lucide-react';

export default function DatasetsPage() {
  const datasets = [
    {
      id: 'firecrawl',
      name: 'Firecrawl',
      description: 'Automatically fetch datasets from Wikipedia, books, and web sources',
      icon: BookOpen,
      color: 'from-blue-500 to-blue-600',
      features: ['Wikipedia articles', 'Web scraping', 'Markdown extraction', 'Auto-processing'],
    },
    {
      id: 'github',
      name: 'GitHub Repositories',
      description: 'Clone and use data from GitHub repositories for training',
      icon: Github,
      color: 'from-purple-500 to-purple-600',
      features: ['Repository cloning', 'Code analysis', 'Documentation parsing', 'Version control'],
    },
    {
      id: 'huggingface',
      name: 'Hugging Face',
      description: 'Access pre-existing datasets from Hugging Face Hub',
      icon: Zap,
      color: 'from-yellow-500 to-yellow-600',
      features: ['Pre-trained datasets', 'Model hub', 'Community datasets', 'Easy integration'],
    },
    {
      id: 'kaggle',
      name: 'Kaggle',
      description: 'Use datasets from Kaggle competitions and datasets',
      icon: Database,
      color: 'from-green-500 to-green-600',
      features: ['Competition data', 'Public datasets', 'Curated collections', 'High quality'],
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="space-y-2 mb-8">
        <h2 className="text-3xl font-bold text-white">Available Datasets</h2>
        <p className="text-slate-400">Choose from multiple data sources to train your models</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {datasets.map((dataset) => {
          const Icon = dataset.icon;
          return (
            <div
              key={dataset.id}
              className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 hover:border-blue-500/50 transition-all hover:bg-slate-700/70"
            >
              <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${dataset.color} mb-4`}>
                <Icon size={24} className="text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">{dataset.name}</h3>
              <p className="text-slate-400 mb-4">{dataset.description}</p>

              <div className="space-y-2">
                <p className="text-sm font-medium text-slate-300">Features:</p>
                <ul className="space-y-1">
                  {dataset.features.map((feature, idx) => (
                    <li key={idx} className="text-sm text-slate-400 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Guide */}
      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-lg p-6 space-y-4">
        <h3 className="text-xl font-semibold text-white">How to Use Datasets</h3>
        <div className="space-y-3 text-slate-300">
          <p>
            <span className="font-semibold text-white">1. Firecrawl:</span> Automatically scrapes Wikipedia and web sources. Great for quick prototyping.
          </p>
          <p>
            <span className="font-semibold text-white">2. GitHub:</span> Clone repositories and use their data. Perfect for code-related models.
          </p>
          <p>
            <span className="font-semibold text-white">3. Hugging Face:</span> Access pre-existing datasets. Best for standard ML tasks.
          </p>
          <p>
            <span className="font-semibold text-white">4. Kaggle:</span> Use competition and public datasets. Ideal for production models.
          </p>
        </div>
      </div>

      {/* Dataset Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Sources', value: '4' },
          { label: 'Supported Formats', value: '10+' },
          { label: 'Max Dataset Size', value: 'Unlimited' },
          { label: 'Processing Speed', value: 'Real-time' },
        ].map((stat, idx) => (
          <div key={idx} className="bg-slate-700/50 border border-slate-600 rounded-lg p-4 text-center">
            <p className="text-slate-400 text-sm mb-1">{stat.label}</p>
            <p className="text-2xl font-bold text-white">{stat.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
