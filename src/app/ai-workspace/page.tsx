'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/lib/supabase';
import styles from './page.module.css';
import { SandboxPreview } from './components/SandboxPreview';
import { startTrainingProcess, pollTrainingStatus, isE2bUrl, isFallbackLocalUrl } from './functions';

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
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

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

  const clearPoll = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => clearPoll();
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

      const startRes = await startTrainingProcess(newEventId, input, projectId, user.id);
      if (!startRes.success) {
        throw new Error(startRes.error || 'Failed to start training');
      }

      // Poll for sandbox URL and status
      clearPoll();
      pollIntervalRef.current = setInterval(async () => {
        try {
          const res = await pollTrainingStatus(newEventId);
          // Expecting shape to include status and maybe sandboxUrl
          if (res?.status) {
            setStatus(String(res.status));
          }
          const url = res?.sandboxUrl || res?.url || res?.appUrl;
          if (isE2bUrl(url) || isFallbackLocalUrl(url)) {
            setSandboxUrl(url);
            setStatus('Sandbox ready');
            clearPoll();
          }
          if (res?.done || res?.success === true) {
            clearPoll();
          }
        } catch (err) {
          // keep polling, but surface a minimal status
          setStatus('Waiting for sandbox...');
        }
      }, 2500);
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
        <div className={styles.logo}>
          zehanxtech
        </div>

        <h1 className={styles.title}>
          AI that builds AI
        </h1>
        <p className={styles.subtitle}>
          Describe the AI you want. We generate code, run it in E2B, train and deploy.
        </p>

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

