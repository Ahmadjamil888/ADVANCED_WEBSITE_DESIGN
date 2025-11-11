'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/lib/supabase';
import styles from './page.module.css';
import { SandboxPreview } from './components/SandboxPreview';
import { startTrainingWithSSE, isE2bUrl, isFallbackLocalUrl } from './functions';
import { AI_MODELS, DEFAULT_MODEL } from '@/lib/ai/models';
import { SignOutButton } from './components/SignOutButton';

interface Project {
  id: string;
  name: string;
  userId: string;
  createdAt: string;
  updatedAt: string;
}

export default function AIWorkspaceLanding() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [input, setInput] = useState('');
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [sandboxUrl, setSandboxUrl] = useState<string | undefined>(undefined);
  const [eventId, setEventId] = useState<string>('');
  const [modelKey, setModelKey] = useState<string>(DEFAULT_MODEL);

  useEffect(() => {
    if (!loading && user) {
      loadProjects();
    }
  }, [loading, user]);

  const loadProjects = async () => {
    if (!user || !supabase) return;

    try {
      const { data, error } = await supabase
        .from('Project')
        .select('*')
        .eq('userId', user.id)
        .order('updatedAt', { ascending: false })
        .limit(6);

      if (!error && data) {
        setProjects(data as unknown as Project[]);
      }
    } catch {
      // Table may not exist in this Supabase instance; ignore for now
    }
  };

  useEffect(() => {
    return () => {};
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user || isLoading || !supabase) return;

    setIsLoading(true);
    setStatus('Creating project...');
    setSandboxUrl(undefined);

    try {
      // Create new project if table exists; otherwise fallback to a local id
      let projectId = '';
      try {
        const { data: project, error } = await (supabase
          .from('Project')
          .insert as any)({
          name: input.slice(0, 50),
          userId: user.id,
        }).select().single();
        if (error) throw error;
        projectId = (project as any).id;
      } catch {
        projectId = crypto.randomUUID();
      }

      // Start AI training/generation pipeline
      const newEventId = crypto.randomUUID();
      setEventId(newEventId);
      setStatus('Starting training and sandbox...');

      // Stream SSE updates from generator
      const startRes = await startTrainingWithSSE(
        input,
        projectId,
        user.id,
        {
          modelKey,
          onStatus: (msg) => setStatus(msg || ''),
          onDeploymentUrl: (url) => {
            if (isE2bUrl(url) || isFallbackLocalUrl(url)) {
              setSandboxUrl(url);
            }
          },
          onError: (msg) => setStatus(`Error: ${msg}`),
        }
      );
      if (!startRes.success) throw new Error(startRes.error || 'Failed to start training');
    } catch (error) {
      console.error('Error creating project:', error);
      setIsLoading(false);
      setStatus('Error starting training');
    }
  };

  const handleProjectClick = (projectId: string) => {
    router.push(`/ai-workspace/${projectId}`);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.spinner} />
        </div>
      </div>
    );
  }

  if (!user) {
    router.push('/login');
    return null;
  }

  if (!supabase) {
    return (
      <div className={styles.container}>
        <div className={styles.content}>
          <p style={{ color: '#fff' }}>
            Supabase is not configured. Please set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div style={{ position: 'absolute', top: 16, right: 16 }}>
          <SignOutButton />
        </div>
        <div className={styles.logo}>
          zehanxtech
        </div>

        <h1 className={styles.title}>
          AI that builds AI
        </h1>
        <p className={styles.subtitle}>
          Describe the AI you want. We generate code, run it in E2B, train and deploy.
        </p>

        {/* Model selector */}
        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginBottom: '1rem' }}>
          <select
            value={modelKey}
            onChange={(e) => setModelKey(e.target.value)}
            style={{
              background: '#000',
              color: '#fff',
              border: '1px solid #fff',
              borderRadius: 0,
              padding: '0.5rem 0.75rem',
            }}
          >
            {Object.entries(AI_MODELS).map(([key, model]) => (
              <option key={key} value={key} style={{ color: '#000' }}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.inputContainer}>
          <form onSubmit={handleSubmit} className={styles.inputWrapper}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g., Build an image classifier for cats vs dogs"
              className={styles.input}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={styles.submitButton}
            >
              {isLoading ? (
                <>Working...</>
              ) : (
                <>Generate & Run</>
              )}
            </button>
          </form>
        </div>

        <div className={styles.suggestions}>
          <button
            onClick={() => handleSuggestionClick('Create a sentiment analysis model using BERT')}
            className={styles.suggestionButton}
          >
            Sentiment Analysis
          </button>
          <button
            onClick={() => handleSuggestionClick('Build an image classifier for cats vs dogs')}
            className={styles.suggestionButton}
          >
            Image Classification
          </button>
          <button
            onClick={() => handleSuggestionClick('Create a text generation model using GPT-2')}
            className={styles.suggestionButton}
          >
            Text Generation
          </button>
          <button
            onClick={() => handleSuggestionClick('Build a recommendation system')}
            className={styles.suggestionButton}
          >
            Recommendation System
          </button>
        </div>

        <div className={styles.statusPanel}>
          <div>Status: {status || 'Idle'}</div>
          {sandboxUrl && (
            <div style={{ marginTop: '0.5rem' }}>
              Sandbox URL: <a href={sandboxUrl} target="_blank" rel="noreferrer" style={{ color: '#fff', textDecoration: 'underline' }}>{sandboxUrl}</a>
            </div>
          )}
        </div>

        <div className={styles.sandboxPanel}>
          <SandboxPreview sandboxUrl={sandboxUrl} />
        </div>

        {/* Recent Projects */}
        {projects.length > 0 && (
          <div className={styles.projectsSection}>
            <div className={styles.projectsHeader}>
              <h2 className={styles.projectsTitle}>My Projects</h2>
              <button className={styles.createProjectButton}>
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ width: 16, height: 16 }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Project
              </button>
            </div>
            <div className={styles.projectsList}>
              {projects.map((project) => (
                <div
                  key={project.id}
                  className={styles.projectCard}
                  onClick={() => handleProjectClick(project.id)}
                >
                  <div className={styles.projectName}>{project.name}</div>
                  <div className={styles.projectDate}>
                    {new Date(project.updatedAt).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

