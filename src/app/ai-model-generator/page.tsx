'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface TrainingStats {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

interface Step {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  details?: string;
}

export default function AIModelGeneratorPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'generator' | 'models' | 'deployments' | 'settings'>('generator');
  const [prompt, setPrompt] = useState('');
  const [modelName, setModelName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useAWS, setUseAWS] = useState(false);
  const [awsKey, setAwsKey] = useState('');
  const [showAWSOption, setShowAWSOption] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [models, setModels] = useState<any[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [steps, setSteps] = useState<Step[]>([
    { name: 'Code Generation', status: 'pending' },
    { name: 'Sandbox Creation', status: 'pending' },
    { name: 'Model Training', status: 'pending' },
    { name: 'E2B Deployment', status: 'pending' },
  ]);
  const [trainingStats, setTrainingStats] = useState<TrainingStats[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [deploymentResult, setDeploymentResult] = useState<any>(null);

  // Fetch user ID and models on mount
  useEffect(() => {
    const fetchUserAndModels = async () => {
      try {
        // Get user from session
        const sessionResponse = await fetch('/api/auth/session');
        if (sessionResponse.ok) {
          const sessionData = await sessionResponse.json();
          if (sessionData.user?.id) {
            setUserId(sessionData.user.id);
            // Fetch user's models
            fetchModels(sessionData.user.id);
          }
        }
      } catch (err) {
        console.error('Error fetching user:', err);
      }
    };
    fetchUserAndModels();
  }, []);

  const fetchModels = async (uid: string) => {
    try {
      setLoadingModels(true);
      const response = await fetch(`/api/models/get-models?userId=${uid}`);
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
      }
    } catch (err) {
      console.error('Error fetching models:', err);
    } finally {
      setLoadingModels(false);
    }
  };

  const updateStep = (index: number, status: 'pending' | 'in-progress' | 'completed' | 'error', details?: string) => {
    setSteps((prev: Step[]) => {
      const newSteps = [...prev];
      newSteps[index] = { ...newSteps[index], status, details };
      return newSteps;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setShowAWSOption(true);
  };

  const handleProceedWithDeployment = async (deploymentType: 'e2b' | 'aws') => {
    setShowAWSOption(false);
    setIsLoading(true);
    setError(null);
    setDeploymentResult(null);
    setTrainingStats([]);
    setSteps([
      { name: 'Code Generation', status: 'in-progress' },
      { name: 'Sandbox Creation', status: 'pending' },
      { name: 'Model Training', status: 'pending' },
      { name: 'Deployment', status: 'pending' },
    ]);

    try {
      console.log('Starting orchestration with prompt:', prompt);

      const response = await fetch('/api/ai/orchestrate-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt, 
          deploymentType,
          userId,
          modelName: modelName || `Model-${Date.now()}`,
          awsKey: deploymentType === 'aws' ? awsKey : undefined 
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate and train model');
      }

      const data = await response.json();
      console.log('Orchestration response:', data);

      updateStep(0, 'completed', 'Code generated successfully');
      updateStep(1, 'completed', `Sandbox: ${data.sandboxId?.slice(0, 8)}...`);
      updateStep(2, 'completed', 'Model trained successfully');
      updateStep(3, 'completed', `Deployed to ${deploymentType.toUpperCase()}`);

      setDeploymentResult({
        deploymentUrl: data.deploymentUrl,
        sandboxId: data.sandboxId,
        modelType: data.modelType,
        deploymentType,
      });
    } catch (err) {
      console.error('Error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);

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
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, overflow: 'hidden' }}>
      <style>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body, html {
          width: 100%;
          height: 100%;
          margin: 0;
          padding: 0;
        }

        .dashboard-wrapper {
          width: 100vw;
          height: 100vh;
          min-height: 100vh;
          background: #000000;
          display: flex;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
          margin: 0;
          padding: 0;
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

        /* Glowing Prompt Box Styles */
        .grid {
          height: 800px;
          width: 800px;
          background-image: linear-gradient(to right, #0f0f10 1px, transparent 1px),
            linear-gradient(to bottom, #0f0f10 1px, transparent 1px);
          background-size: 1rem 1rem;
          background-position: center center;
          position: absolute;
          z-index: -1;
          filter: blur(1px);
        }

        .white, .border, .darkBorderBg, .glow {
          max-height: 70px;
          max-width: 314px;
          height: 100%;
          width: 100%;
          position: absolute;
          overflow: hidden;
          z-index: -1;
          border-radius: 12px;
          filter: blur(3px);
        }

        .input {
          background-color: #010201;
          border: none;
          width: 100%;
          height: auto;
          min-height: 120px;
          border-radius: 10px;
          color: white;
          padding: 20px;
          font-size: 16px;
          font-family: inherit;
          resize: vertical;
        }

        #poda {
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          margin-bottom: 1.5rem;
        }

        .input::placeholder {
          color: #c0b9c0;
        }

        .input:focus {
          outline: none;
        }

        #main:focus-within > #input-mask {
          display: none;
        }

        #input-mask {
          pointer-events: none;
          width: 100px;
          height: 20px;
          position: absolute;
          background: linear-gradient(90deg, transparent, black);
          top: 18px;
          left: 70px;
        }

        #pink-mask {
          pointer-events: none;
          width: 30px;
          height: 20px;
          position: absolute;
          background: #cf30aa;
          top: 10px;
          left: 5px;
          filter: blur(20px);
          opacity: 0.8;
          transition: all 2s;
        }

        #main:hover > #pink-mask {
          opacity: 0;
        }

        .white {
          max-height: 63px;
          max-width: 307px;
          border-radius: 10px;
          filter: blur(2px);
        }

        .white::before {
          content: "";
          z-index: -2;
          text-align: center;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) rotate(83deg);
          position: absolute;
          width: 600px;
          height: 600px;
          background-repeat: no-repeat;
          background-position: 0 0;
          filter: brightness(1.4);
          background-image: conic-gradient(
            rgba(0, 0, 0, 0) 0%,
            #a099d8,
            rgba(0, 0, 0, 0) 8%,
            rgba(0, 0, 0, 0) 50%,
            #dfa2da,
            rgba(0, 0, 0, 0) 58%
          );
          transition: all 2s;
        }

        .border {
          max-height: 59px;
          max-width: 303px;
          border-radius: 11px;
          filter: blur(0.5px);
        }

        .border::before {
          content: "";
          z-index: -2;
          text-align: center;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) rotate(70deg);
          position: absolute;
          width: 600px;
          height: 600px;
          filter: brightness(1.3);
          background-repeat: no-repeat;
          background-position: 0 0;
          background-image: conic-gradient(
            #1c191c,
            #402fb5 5%,
            #1c191c 14%,
            #1c191c 50%,
            #cf30aa 60%,
            #1c191c 64%
          );
          transition: all 2s;
        }

        .darkBorderBg {
          max-height: 65px;
          max-width: 312px;
        }

        .darkBorderBg::before {
          content: "";
          z-index: -2;
          text-align: center;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) rotate(82deg);
          position: absolute;
          width: 600px;
          height: 600px;
          background-repeat: no-repeat;
          background-position: 0 0;
          background-image: conic-gradient(
            rgba(0, 0, 0, 0),
            #18116a,
            rgba(0, 0, 0, 0) 10%,
            rgba(0, 0, 0, 0) 50%,
            #6e1b60,
            rgba(0, 0, 0, 0) 60%
          );
          transition: all 2s;
        }

        #poda:hover > .darkBorderBg::before {
          transform: translate(-50%, -50%) rotate(-98deg);
        }

        #poda:hover > .glow::before {
          transform: translate(-50%, -50%) rotate(-120deg);
        }

        #poda:hover > .white::before {
          transform: translate(-50%, -50%) rotate(-97deg);
        }

        #poda:hover > .border::before {
          transform: translate(-50%, -50%) rotate(-110deg);
        }

        #poda:focus-within > .darkBorderBg::before {
          transform: translate(-50%, -50%) rotate(442deg);
          transition: all 4s;
        }

        #poda:focus-within > .glow::before {
          transform: translate(-50%, -50%) rotate(420deg);
          transition: all 4s;
        }

        #poda:focus-within > .white::before {
          transform: translate(-50%, -50%) rotate(443deg);
          transition: all 4s;
        }

        #poda:focus-within > .border::before {
          transform: translate(-50%, -50%) rotate(430deg);
          transition: all 4s;
        }

        .glow {
          overflow: hidden;
          filter: blur(30px);
          opacity: 0.4;
          max-height: 130px;
          max-width: 354px;
        }

        .glow:before {
          content: "";
          z-index: -2;
          text-align: center;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) rotate(60deg);
          position: absolute;
          width: 999px;
          height: 999px;
          background-repeat: no-repeat;
          background-position: 0 0;
          background-image: conic-gradient(
            #000,
            #402fb5 5%,
            #000 38%,
            #000 50%,
            #cf30aa 60%,
            #000 87%
          );
          transition: all 2s;
        }

        #filter-icon {
          position: absolute;
          top: 8px;
          right: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2;
          max-height: 40px;
          max-width: 38px;
          height: 100%;
          width: 100%;
          isolation: isolate;
          overflow: hidden;
          border-radius: 10px;
          background: linear-gradient(180deg, #161329, black, #1d1b4b);
          border: 1px solid transparent;
        }

        .filterBorder {
          height: 42px;
          width: 40px;
          position: absolute;
          overflow: hidden;
          top: 7px;
          right: 7px;
          border-radius: 10px;
        }

        .filterBorder::before {
          content: "";
          text-align: center;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) rotate(90deg);
          position: absolute;
          width: 600px;
          height: 600px;
          background-repeat: no-repeat;
          background-position: 0 0;
          filter: brightness(1.35);
          background-image: conic-gradient(
            rgba(0, 0, 0, 0),
            #3d3a4f,
            rgba(0, 0, 0, 0) 50%,
            rgba(0, 0, 0, 0) 50%,
            #3d3a4f,
            rgba(0, 0, 0, 0) 100%
          );
        }

        #main {
          position: relative;
          width: 100%;
        }

        #search-icon {
          position: absolute;
          left: 20px;
          top: 15px;
          color: #999;
        }

        /* Prompt Section */
        .prompt-section {
          margin-bottom: 2rem;
        }

        .prompt-section h2 {
          font-size: 1.75rem;
          color: #ffffff;
          margin-bottom: 0.5rem;
        }

        .prompt-section > p {
          color: #999999;
          margin-bottom: 1.5rem;
        }

        .submit-btn {
          background: linear-gradient(135deg, #402fb5, #cf30aa);
          color: white;
          border: none;
          padding: 12px 32px;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          font-size: 1rem;
          transition: all 0.3s ease;
          margin-top: 1rem;
        }

        .submit-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 10px 30px rgba(207, 48, 170, 0.3);
        }

        .submit-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Deployment Modal */
        .deployment-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .deployment-modal-content {
          background: #0a0a0a;
          border: 1px solid #222222;
          border-radius: 12px;
          padding: 2rem;
          max-width: 500px;
          width: 90%;
        }

        .deployment-modal-content h2 {
          color: #ffffff;
          margin-bottom: 0.5rem;
        }

        .deployment-modal-content > p {
          color: #999999;
          margin-bottom: 1.5rem;
        }

        .deployment-options {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .deployment-option {
          background: #111111;
          border: 1px solid #222222;
          border-radius: 8px;
          padding: 1.5rem;
          cursor: pointer;
          transition: all 0.3s ease;
          text-align: center;
          color: #ffffff;
          font-family: inherit;
        }

        .deployment-option:hover:not(:disabled) {
          border-color: #cf30aa;
          background: #1a0a1a;
        }

        .deployment-option:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .option-icon {
          font-size: 2rem;
          margin-bottom: 0.5rem;
        }

        .option-title {
          font-weight: 600;
          margin-bottom: 0.25rem;
        }

        .option-desc {
          font-size: 0.85rem;
          color: #999999;
        }

        .aws-input-section {
          background: #111111;
          border: 1px solid #222222;
          border-radius: 8px;
          padding: 1rem;
          margin-bottom: 1rem;
        }

        .aws-input-section label {
          display: block;
          color: #ffffff;
          margin-bottom: 0.5rem;
          font-size: 0.9rem;
        }

        .aws-input-section input {
          width: 100%;
          padding: 0.75rem;
          background: #0a0a0a;
          border: 1px solid #333333;
          border-radius: 6px;
          color: #ffffff;
          margin-bottom: 1rem;
          font-family: inherit;
        }

        .cancel-btn {
          width: 100%;
          background: transparent;
          border: 1px solid #333333;
          color: #ffffff;
          padding: 0.75rem;
          border-radius: 6px;
          cursor: pointer;
          font-family: inherit;
          transition: all 0.3s ease;
        }

        .cancel-btn:hover:not(:disabled) {
          border-color: #666666;
          background: #111111;
        }

        /* Training Stats */
        .training-stats-section {
          background: #0a0a0a;
          border: 1px solid #222222;
          border-radius: 8px;
          padding: 1.5rem;
          margin-top: 2rem;
        }

        .training-stats-section h3 {
          color: #ffffff;
          margin-bottom: 1rem;
        }

        .steps-container {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .step-item {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: #111111;
          border-radius: 6px;
          border-left: 3px solid #333333;
        }

        .step-item.step-completed {
          border-left-color: #10b981;
          background: rgba(16, 185, 129, 0.05);
        }

        .step-item.step-in-progress {
          border-left-color: #3b82f6;
          background: rgba(59, 130, 246, 0.05);
        }

        .step-item.step-error {
          border-left-color: #ef4444;
          background: rgba(239, 68, 68, 0.05);
        }

        .step-number {
          min-width: 32px;
          height: 32px;
          border-radius: 50%;
          background: #222222;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-weight: 600;
        }

        .step-content {
          flex: 1;
        }

        .step-name {
          color: #ffffff;
          font-weight: 500;
        }

        .step-details {
          color: #999999;
          font-size: 0.85rem;
          margin-top: 0.25rem;
        }

        .step-status {
          color: #999999;
          font-size: 1.25rem;
        }

        .stats-table {
          background: #111111;
          border-radius: 6px;
          overflow: hidden;
        }

        .stats-table h4 {
          color: #ffffff;
          padding: 1rem;
          border-bottom: 1px solid #222222;
          margin: 0;
        }

        .stats-table table {
          width: 100%;
          border-collapse: collapse;
        }

        .stats-table th {
          background: #0a0a0a;
          color: #999999;
          padding: 0.75rem;
          text-align: left;
          font-size: 0.85rem;
          font-weight: 600;
          border-bottom: 1px solid #222222;
        }

        .stats-table td {
          padding: 0.75rem;
          color: #ffffff;
          border-bottom: 1px solid #1a1a1a;
        }

        .stats-table tr:last-child td {
          border-bottom: none;
        }

        /* Error Card */
        .error-card {
          background: #1a0a0a;
          border: 1px solid #4a1a1a;
          border-radius: 8px;
          padding: 1.5rem;
          margin-top: 1.5rem;
        }

        .error-card h3 {
          color: #ff6b6b;
          margin-bottom: 0.5rem;
        }

        .error-card p {
          color: #ff9999;
          margin-bottom: 1rem;
        }

        .retry-btn {
          background: #ff6b6b;
          color: white;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 600;
          font-family: inherit;
          transition: all 0.3s ease;
        }

        .retry-btn:hover {
          background: #ff5252;
        }

        /* Deployment Result */
        .deployment-result-card {
          background: #0a0a0a;
          border: 1px solid #222222;
          border-radius: 8px;
          padding: 2rem;
          text-align: center;
        }

        .deployment-result-card h2 {
          color: #10b981;
          margin-bottom: 1.5rem;
        }

        .result-details {
          background: #111111;
          border-radius: 6px;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
          text-align: left;
        }

        .result-item {
          display: flex;
          flex-direction: column;
          margin-bottom: 1rem;
        }

        .result-item:last-child {
          margin-bottom: 0;
        }

        .result-item .label {
          color: #999999;
          font-size: 0.85rem;
          margin-bottom: 0.25rem;
        }

        .result-item code {
          background: #0a0a0a;
          border: 1px solid #222222;
          border-radius: 4px;
          padding: 0.5rem;
          color: #10b981;
          font-family: 'Courier New', monospace;
          word-break: break-all;
        }

        .new-model-btn {
          background: linear-gradient(135deg, #402fb5, #cf30aa);
          color: white;
          border: none;
          padding: 0.75rem 2rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 600;
          font-family: inherit;
          transition: all 0.3s ease;
        }

        .new-model-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 30px rgba(207, 48, 170, 0.3);
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

          .deployment-options {
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
              className={`dashboard-nav-item ${activeTab === 'deployments' ? 'active' : ''}`}
              onClick={() => setActiveTab('deployments')}
            >
              Deployments
            </button>
            <button
              className={`dashboard-nav-item ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              Settings
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
              {activeTab === 'deployments' && 'Deployments'}
              {activeTab === 'settings' && 'Settings'}
            </h1>
          </div>

          <div className="dashboard-content">
            {/* Generator Tab */}
            {activeTab === 'generator' && (
              <div style={{ padding: '2rem' }}>
                {showAWSOption && !deploymentResult ? (
                  <div className="deployment-modal">
                    <div className="deployment-modal-content">
                      <h2>Choose Deployment Option</h2>
                      <p>Where would you like to deploy your model?</p>
                      
                      <div className="deployment-options">
                        <button 
                          className="deployment-option e2b-option"
                          onClick={() => handleProceedWithDeployment('e2b')}
                          disabled={isLoading}
                        >
                          <div className="option-icon">☁️</div>
                          <div className="option-title">E2B Sandbox</div>
                          <div className="option-desc">Fast, serverless deployment</div>
                        </button>
                        
                        <button 
                          className="deployment-option aws-option"
                          onClick={() => setUseAWS(!useAWS)}
                        >
                          <div className="option-icon">🔑</div>
                          <div className="option-title">AWS</div>
                          <div className="option-desc">Deploy to your AWS account</div>
                        </button>
                      </div>

                      {useAWS && (
                        <div className="aws-input-section">
                          <label>AWS Access Key ID</label>
                          <input 
                            type="password" 
                            value={awsKey} 
                            onChange={(e) => setAwsKey(e.target.value)}
                            placeholder="Enter your AWS key"
                          />
                          <button 
                            className="deployment-option aws-option"
                            onClick={() => handleProceedWithDeployment('aws')}
                            disabled={!awsKey || isLoading}
                          >
                            Deploy to AWS
                          </button>
                        </div>
                      )}

                      <button 
                        className="cancel-btn"
                        onClick={() => setShowAWSOption(false)}
                        disabled={isLoading}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : !deploymentResult ? (
                  <div>
                    <div className="prompt-section">
                      <h2>Describe Your AI Model</h2>
                      <p>Tell us what kind of model you want to create</p>
                      
                      <form onSubmit={handleSubmit}>
                        <div style={{ marginBottom: '1.5rem' }}>
                          <label style={{ display: 'block', color: '#ffffff', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Model Name (Optional)</label>
                          <input 
                            type="text"
                            placeholder="e.g., Sentiment Analyzer"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            disabled={isLoading}
                            style={{
                              width: '100%',
                              padding: '0.75rem',
                              background: '#0a0a0a',
                              border: '1px solid #222222',
                              borderRadius: '6px',
                              color: '#ffffff',
                              fontFamily: 'inherit',
                              fontSize: '0.9rem'
                            }}
                          />
                        </div>
                        <div id="poda">
                          <div className="glow"></div>
                          <div className="darkBorderBg"></div>
                          <div className="darkBorderBg"></div>
                          <div className="darkBorderBg"></div>
                          <div className="white"></div>
                          <div className="border"></div>
                          <div id="main">
                            <textarea 
                              placeholder="E.g., Create a sentiment analysis model that classifies text as positive, negative, or neutral..."
                              className="input"
                              value={prompt}
                              onChange={(e) => setPrompt(e.target.value)}
                              disabled={isLoading}
                              style={{ resize: 'vertical', minHeight: '120px' }}
                            />
                            <div id="input-mask"></div>
                            <div id="pink-mask"></div>
                            <div className="filterBorder"></div>
                            <div id="filter-icon">
                              <svg preserveAspectRatio="none" height="27" width="27" viewBox="4.8 4.56 14.832 15.408" fill="none">
                                <path d="M8.16 6.65002H15.83C16.47 6.65002 16.99 7.17002 16.99 7.81002V9.09002C16.99 9.56002 16.7 10.14 16.41 10.43L13.91 12.64C13.56 12.93 13.33 13.51 13.33 13.98V16.48C13.33 16.83 13.1 17.29 12.81 17.47L12 17.98C11.24 18.45 10.2 17.92 10.2 16.99V13.91C10.2 13.5 9.97 12.98 9.73 12.69L7.52 10.36C7.23 10.08 7 9.55002 7 9.20002V7.87002C7 7.17002 7.52 6.65002 8.16 6.65002Z" stroke="#d6d6e6" strokeWidth="1" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"></path>
                              </svg>
                            </div>
                            <div id="search-icon">
                              <svg xmlns="http://www.w3.org/2000/svg" width="24" viewBox="0 0 24 24" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" height="24" fill="none" className="feather feather-search">
                                <circle stroke="url(#search)" r="8" cy="11" cx="11"></circle>
                                <line stroke="url(#searchl)" y2="16.65" y1="22" x2="16.65" x1="22"></line>
                                <defs>
                                  <linearGradient gradientTransform="rotate(50)" id="search">
                                    <stop stopColor="#f8e7f8" offset="0%"></stop>
                                    <stop stopColor="#b6a9b7" offset="50%"></stop>
                                  </linearGradient>
                                  <linearGradient id="searchl">
                                    <stop stopColor="#b6a9b7" offset="0%"></stop>
                                    <stop stopColor="#837484" offset="50%"></stop>
                                  </linearGradient>
                                </defs>
                              </svg>
                            </div>
                          </div>
                        </div>
                        <div className="grid"></div>
                        
                        <button 
                          type="submit" 
                          className="submit-btn"
                          disabled={!prompt.trim() || isLoading}
                        >
                          {isLoading ? 'Processing...' : 'Generate Model'}
                        </button>
                      </form>
                    </div>

                    {isLoading && (
                      <div className="training-stats-section">
                        <h3>Training Progress</h3>
                        <div className="steps-container">
                          {steps.map((step, idx) => (
                            <div key={idx} className={`step-item step-${step.status}`}>
                              <div className="step-number">{idx + 1}</div>
                              <div className="step-content">
                                <div className="step-name">{step.name}</div>
                                {step.details && <div className="step-details">{step.details}</div>}
                              </div>
                              <div className="step-status">
                                {step.status === 'completed' && '✓'}
                                {step.status === 'in-progress' && '⟳'}
                                {step.status === 'error' && '✕'}
                              </div>
                            </div>
                          ))}
                        </div>

                        {trainingStats.length > 0 && (
                          <div className="stats-table">
                            <h4>Epoch Statistics</h4>
                            <table>
                              <thead>
                                <tr>
                                  <th>Epoch</th>
                                  <th>Loss</th>
                                  <th>Accuracy</th>
                                  <th>Val Loss</th>
                                  <th>Val Accuracy</th>
                                </tr>
                              </thead>
                              <tbody>
                                {trainingStats.map((stat, idx) => (
                                  <tr key={idx}>
                                    <td>{stat.epoch}</td>
                                    <td>{stat.loss.toFixed(4)}</td>
                                    <td>{(stat.accuracy * 100).toFixed(2)}%</td>
                                    <td>{stat.valLoss.toFixed(4)}</td>
                                    <td>{(stat.valAccuracy * 100).toFixed(2)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    )}

                    {error && !isLoading && (
                      <div className="error-card">
                        <h3>Error</h3>
                        <p>{error}</p>
                        <button
                          onClick={() => {
                            setError(null);
                            setSteps([
                              { name: 'Code Generation', status: 'pending' },
                              { name: 'Sandbox Creation', status: 'pending' },
                              { name: 'Model Training', status: 'pending' },
                              { name: 'Deployment', status: 'pending' },
                            ]);
                          }}
                          className="retry-btn"
                        >
                          Try Again
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="deployment-result-card">
                    <h2>✓ Model Deployed Successfully!</h2>
                    <div className="result-details">
                      <div className="result-item">
                        <span className="label">Deployment URL:</span>
                        <code>{deploymentResult.deploymentUrl}</code>
                      </div>
                      <div className="result-item">
                        <span className="label">Sandbox ID:</span>
                        <code>{deploymentResult.sandboxId}</code>
                      </div>
                      <div className="result-item">
                        <span className="label">Model Type:</span>
                        <span>{deploymentResult.modelType}</span>
                      </div>
                      <div className="result-item">
                        <span className="label">Deployment Type:</span>
                        <span>{deploymentResult.deploymentType?.toUpperCase()}</span>
                      </div>
                    </div>
                    <button 
                      onClick={() => {
                        setDeploymentResult(null);
                        setPrompt('');
                        setSteps([
                          { name: 'Code Generation', status: 'pending' },
                          { name: 'Sandbox Creation', status: 'pending' },
                          { name: 'Model Training', status: 'pending' },
                          { name: 'Deployment', status: 'pending' },
                        ]);
                      }}
                      className="new-model-btn"
                    >
                      Create Another Model
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* My Models Tab */}
            {activeTab === 'models' && (
              <div style={{ padding: '2rem' }}>
                {loadingModels ? (
                  <div className="dashboard-card">
                    <div className="dashboard-card-title">Loading Models...</div>
                  </div>
                ) : models.length === 0 ? (
                  <div className="dashboard-card">
                    <div className="dashboard-card-title">Your Models</div>
                    <p style={{ color: '#666666', fontSize: '0.9rem' }}>
                      No models created yet. Start by generating your first model.
                    </p>
                  </div>
                ) : (
                  <div>
                    <div style={{ marginBottom: '2rem' }}>
                      <h3 style={{ color: '#ffffff', marginBottom: '1rem' }}>Your AI Models ({models.length})</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
                        {models.map((model: any) => (
                          <div key={model.id} className="dashboard-card" style={{ display: 'flex', flexDirection: 'column' }}>
                            <div className="dashboard-card-title">{model.name}</div>
                            <div style={{ color: '#999999', fontSize: '0.85rem', marginBottom: '1rem' }}>
                              <p><strong>Type:</strong> {model.type}</p>
                              <p><strong>Status:</strong> <span style={{ color: '#10b981' }}>{model.status}</span></p>
                              <p><strong>Created:</strong> {new Date(model.created_at).toLocaleDateString()}</p>
                            </div>
                            {model.deployment_url && (
                              <div style={{ marginBottom: '1rem', padding: '0.75rem', background: '#111111', borderRadius: '6px' }}>
                                <p style={{ color: '#999999', fontSize: '0.75rem', marginBottom: '0.25rem' }}>Deployment URL:</p>
                                <code style={{ color: '#10b981', fontSize: '0.8rem', wordBreak: 'break-all' }}>{model.deployment_url}</code>
                              </div>
                            )}
                            <div style={{ marginTop: 'auto', display: 'flex', gap: '0.5rem' }}>
                              <button 
                                onClick={() => window.open(model.deployment_url, '_blank')}
                                disabled={!model.deployment_url}
                                style={{
                                  flex: 1,
                                  padding: '0.5rem',
                                  background: model.deployment_url ? '#10b981' : '#333333',
                                  color: '#ffffff',
                                  border: 'none',
                                  borderRadius: '4px',
                                  cursor: model.deployment_url ? 'pointer' : 'not-allowed',
                                  fontSize: '0.85rem',
                                  fontFamily: 'inherit'
                                }}
                              >
                                View API
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Deployments Tab */}
            {activeTab === 'deployments' && (
              <div style={{ padding: '2rem' }}>
                <div className="dashboard-card">
                  <div className="dashboard-card-title">Active Deployments</div>
                  <p style={{ color: '#666666', fontSize: '0.9rem' }}>
                    No active deployments. Create your first model to see it here.
                  </p>
                </div>
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div style={{ padding: '2rem' }}>
                <div className="dashboard-card">
                  <div className="dashboard-card-title">Account Settings</div>
                  <div style={{ color: '#999999', fontSize: '0.9rem' }}>
                    <p>Settings coming soon...</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
