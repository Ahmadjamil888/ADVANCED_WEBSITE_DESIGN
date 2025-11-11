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
      // Create a project then redirect to the split-screen workspace
      const { data: project, error } = await (supabase
        .from('Project')
        .insert as any)({
        name: input.slice(0, 80),
        userId: user.id,
      }).select().single();
      if (error) throw error;
      router.push(`/ai-workspace/${(project as any).id}?prompt=${encodeURIComponent(input)}`);
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
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className={styles.heroBar}>
          <div className={styles.brand}>zehanxtech</div>
          <div className={styles.heroActions}>
            <select value={modelKey} onChange={(e) => setModelKey(e.target.value)} className={styles.modelSelect}>
              {Object.entries(AI_MODELS).map(([key, model]) => (
                <option key={key} value={key}>{model.name}</option>
              ))}
            </select>
            <SignOutButton />
          </div>
        </div>
        <h1 className={styles.heroTitle}>Create your own universe</h1>
        <p className={styles.heroSubtitle}>Describe anything — AI model, Next.js app, scripts — we’ll build and run it live.</p>
        <form onSubmit={handleSubmit} className={styles.heroForm}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g., Generate a Next.js SaaS with auth and billing"
            className={styles.heroInput}
            disabled={isLoading}
          />
          <button type="submit" disabled={!input.trim() || isLoading} className={styles.heroButton}>
            {isLoading ? 'Working...' : 'Launch'}
          </button>
        </form>
        <div className={styles.quickRow}>
          <button onClick={() => handleSuggestionClick('Create a sentiment analysis model using BERT')} className={styles.quickButton}>Sentiment Analysis</button>
          <button onClick={() => handleSuggestionClick('Build an image classifier for cats vs dogs')} className={styles.quickButton}>Image Classification</button>
          <button onClick={() => handleSuggestionClick('Generate a Next.js blog with MDX and dark mode')} className={styles.quickButton}>Next.js Blog</button>
          <button onClick={() => handleSuggestionClick('Create a recommendation system API with FastAPI')} className={styles.quickButton}>Recommendation API</button>
        </div>
      </div>
      <div className={styles.history}>
        <div className={styles.historyHead}>
          <h2 className={styles.historyTitle}>Previous Projects</h2>
        </div>
        <div className={styles.projectsGrid}>
          {projects.length === 0 ? (
            <div className={styles.emptyHistory}>No projects yet.</div>
          ) : (
            projects.map((project) => (
              <button key={project.id} className={styles.historyCard} onClick={() => handleProjectClick(project.id)}>
                <div className={styles.historyName}>{project.name}</div>
                <div className={styles.historyDate}>{new Date(project.updatedAt).toLocaleString()}</div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

