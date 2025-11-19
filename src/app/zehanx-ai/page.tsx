'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function ZehanxAIDashboard() {
  const [stats, setStats] = useState({ totalModels: 0 });
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetch('/api/models');
        const data = await response.json();
        setStats({ totalModels: data.models?.length || 0 });
      } catch (error) {
        console.error('Failed to load stats:', error);
      }
    };
    loadStats();
  }, []);

  const handlePromptSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      const response = await fetch('/api/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Model_${Date.now()}`,
          description: prompt,
          modelType: 'custom',
          datasetSource: 'firecrawl',
          epochs: 10,
          batchSize: 32,
          learningRate: 0.001,
        }),
      });

      if (response.ok) {
        router.push('/zehanx-ai/generator');
      }
    } catch (error) {
      console.error('Error starting training:', error);
    } finally {
      setIsGenerating(false);
      setPrompt('');
    }
  };

  return (
    <div className="p-6 space-y-8">
      <div className="space-y-4">
        <h2 className="text-3xl font-bold text-white">Welcome to Zehanx AI</h2>
        <p className="text-slate-400 text-lg">
          Your complete AI model generation platform. Create, train, and deploy custom AI models with ease.
        </p>
      </div>

      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-4">Create AI Model</h3>
        <p className="text-slate-400 text-sm mb-4">
          Describe the AI model you want to create. Our system will crawl datasets, generate code with Groq, and train your model.
        </p>
        <form onSubmit={handlePromptSubmit} className="flex gap-3">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Create a sentiment analysis model for Twitter data..."
            className="flex-1 px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            disabled={isGenerating}
          />
          <button
            type="submit"
            disabled={isGenerating || !prompt.trim()}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white font-semibold rounded-lg transition-all flex items-center gap-2"
          >
            {isGenerating ? 'Starting...' : 'Generate'}
          </button>
        </form>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6">
          <p className="text-slate-400 text-sm">Total Models</p>
          <p className="text-3xl font-bold text-white mt-2">{stats.totalModels}</p>
        </div>

        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6">
          <p className="text-slate-400 text-sm">Datasets Available</p>
          <p className="text-3xl font-bold text-white mt-2">4</p>
        </div>

        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6">
          <p className="text-slate-400 text-sm">Training Status</p>
          <p className="text-3xl font-bold text-white mt-2">Ready</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link href="/zehanx-ai/generator" className="group">
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 hover:border-blue-500/50 transition-all hover:bg-slate-700/70 h-full">
            <h3 className="text-xl font-semibold text-white mb-2">Model Generator</h3>
            <p className="text-slate-400 mb-4">Create and train custom AI models with multiple architectures</p>
            <div className="flex items-center text-blue-400">Get Started</div>
          </div>
        </Link>

        <Link href="/zehanx-ai/datasets" className="group">
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 hover:border-blue-500/50 transition-all hover:bg-slate-700/70 h-full">
            <h3 className="text-xl font-semibold text-white mb-2">Datasets</h3>
            <p className="text-slate-400 mb-4">Manage and explore your training datasets from multiple sources</p>
            <div className="flex items-center text-blue-400">Get Started</div>
          </div>
        </Link>

        <Link href="/zehanx-ai/models" className="group">
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-6 hover:border-blue-500/50 transition-all hover:bg-slate-700/70 h-full">
            <h3 className="text-xl font-semibold text-white mb-2">My Models</h3>
            <p className="text-slate-400 mb-4">View, download, and manage your trained AI models</p>
            <div className="flex items-center text-blue-400">Get Started</div>
          </div>
        </Link>
      </div>

      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-3">Quick Start</h3>
        <ol className="space-y-2 text-slate-300">
          <li>1. Use the prompt box above to describe your AI model</li>
          <li>2. System crawls datasets using Firecrawl</li>
          <li>3. Groq generates optimized training code</li>
          <li>4. E2B executes training and displays real-time stats</li>
          <li>5. Download your trained model from My Models</li>
        </ol>
      </div>

      <div className="space-y-4">
        <h3 className="text-xl font-semibold text-white">Platform Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="text-slate-300">✓ Firecrawl Integration - Automatic dataset fetching</div>
          <div className="text-slate-300">✓ Groq API - Code generation for model training</div>
          <div className="text-slate-300">✓ E2B Sandbox - Secure code execution</div>
          <div className="text-slate-300">✓ Real-time Training Statistics - Live monitoring</div>
          <div className="text-slate-300">✓ Multiple Model Architectures - Transformer, LSTM, CNN</div>
          <div className="text-slate-300">✓ Supabase Integration - User data persistence</div>
        </div>
      </div>
    </div>
  );
}
