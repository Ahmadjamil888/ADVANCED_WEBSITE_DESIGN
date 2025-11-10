'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { createClient } from '@supabase/supabase-js';
import styles from './workspace.module.css';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

interface Project {
  id: string;
  name: string;
  userId: string;
  createdAt: string;
  updatedAt: string;
}

export default function AIWorkspaceLanding() {
  const router = useRouter();
  const { user, isLoaded } = useUser();
  const [input, setInput] = useState('');
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isLoaded && user) {
      loadProjects();
    }
  }, [isLoaded, user]);

  const loadProjects = async () => {
    if (!user) return;

    const { data, error } = await supabase
      .from('Project')
      .select('*')
      .eq('userId', user.id)
      .order('updatedAt', { ascending: false })
      .limit(6);

    if (!error && data) {
      setProjects(data);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user || isLoading) return;

    setIsLoading(true);

    try {
      // Create new project
      const { data: project, error } = await supabase
        .from('Project')
        .insert({
          name: input.slice(0, 50),
          userId: user.id,
        })
        .select()
        .single();

      if (error) throw error;

      // Redirect to workspace with project ID
      router.push(`/ai-workspace/${project.id}?prompt=${encodeURIComponent(input)}`);
    } catch (error) {
      console.error('Error creating project:', error);
      setIsLoading(false);
    }
  };

  const handleProjectClick = (projectId: string) => {
    router.push(`/ai-workspace/${projectId}`);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  if (!isLoaded) {
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

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        {/* Logo */}
        <div className={styles.logo}>
          <div className={styles.logoIcon}>⚡</div>
          zehanxtech
        </div>

        {/* Title */}
        <h1 className={styles.title}>
          Build something <span style={{ background: 'linear-gradient(135deg, #fff 0%, #f0f0f0 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Lovable</span>
        </h1>
        <p className={styles.subtitle}>
          Turn your ideas into AI models. Describe what you want to build, and our AI will create, train, and deploy it for you.
        </p>

        {/* Input */}
        <div className={styles.inputContainer}>
          <form onSubmit={handleSubmit} className={styles.inputWrapper}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="What do you want to build today?"
              className={styles.input}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={styles.submitButton}
            >
              {isLoading ? (
                <>
                  <div className={styles.spinner} />
                  Creating...
                </>
              ) : (
                <>
                  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Create
                </>
              )}
            </button>
          </form>
        </div>

        {/* Suggestions */}
        <div className={styles.suggestions}>
          <button
            onClick={() => handleSuggestionClick('Create a sentiment analysis model using BERT')}
            className={styles.suggestionButton}
          >
            💬 Sentiment Analysis
          </button>
          <button
            onClick={() => handleSuggestionClick('Build an image classifier for cats vs dogs')}
            className={styles.suggestionButton}
          >
            🖼️ Image Classification
          </button>
          <button
            onClick={() => handleSuggestionClick('Create a text generation model using GPT-2')}
            className={styles.suggestionButton}
          >
            ✍️ Text Generation
          </button>
          <button
            onClick={() => handleSuggestionClick('Build a recommendation system')}
            className={styles.suggestionButton}
          >
            🎯 Recommendation System
          </button>
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
