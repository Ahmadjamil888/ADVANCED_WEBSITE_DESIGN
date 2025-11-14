'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/lib/supabase';
import { AI_MODELS, DEFAULT_MODEL } from '@/lib/ai/models';
import { SignOutButton } from '../components/SignOutButton';
import { DeploymentOptions } from '../components/DeploymentOptions';
import styles from './page.module.css';

interface Message {
  id: string;
  content: string;
  role: 'USER' | 'ASSISTANT';
  type: 'RESULT' | 'ERROR';
  projectId: string;
  createdAt: string;
}

interface Project {
  id: string;
  name: string;
  userId: string;
  createdAt: string;
  updatedAt: string;
}

export default function WorkspacePage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const { user, loading } = useAuth();
  
  const projectId = params.id as string;
  const initialPrompt = searchParams.get('prompt');

  const [project, setProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState<'code' | 'sandbox' | 'deploy'>('code');
  const [generatedFiles, setGeneratedFiles] = useState<Record<string, string>>({});
  const [sandboxUrl, setSandboxUrl] = useState<string>();
  const [modelKey, setModelKey] = useState<string>(DEFAULT_MODEL);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!loading && user) {
      loadProject();
      loadProjects();
      loadMessages();
    }
  }, [loading, user, projectId]);

  useEffect(() => {
    if (initialPrompt && messages.length === 0 && !isGenerating) {
      setInput(initialPrompt);
      // Auto-submit initial prompt
      setTimeout(() => {
        handleSubmit(new Event('submit') as any);
      }, 500);
    }
  }, [initialPrompt, messages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  };

  const loadProject = async () => {
    if (!supabase) return;
    
    const { data, error } = await supabase
      .from('ai_models')
      .select('*')
      .eq('id', projectId)
      .single();

    if (!error && data) {
      setProject(data as any);
    }
  };

  const loadProjects = async () => {
    if (!user || !supabase) return;

    const { data, error } = await supabase
      .from('ai_models')
      .select('*')
      .eq('user_id', user.id)
      .order('updated_at', { ascending: false });

    if (!error && data) {
      setProjects(data as any);
    }
  };

  const loadMessages = async () => {
    if (!supabase) return;
    
    const { data, error } = await supabase
      .from('messages')
      .select('*')
      .eq('chat_id', projectId)
      .order('created_at', { ascending: true });

    if (!error && data) {
      setMessages(data as any);
    }
  };

  const parseFilesFromContent = (content: string): Record<string, string> => {
    const files: Record<string, string> = {};
    const fileRegex = /<file path="([^"]+)">([\s\S]*?)(?:<\/file>|$)/g;
    let match;
    
    while ((match = fileRegex.exec(content)) !== null) {
      let [, path, fileContent] = match;
      if (path === 'requirements' || path === 'requirement') {
        path = 'requirements.txt';
      }
      if ((path === 'train' || path === 'app') && !path.endsWith('.py')) {
        path = path + '.py';
      }
      files[path] = fileContent.trim();
    }
    
    return files;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating || !user || !supabase) return;

    const userMessageContent = input.trim();
    setInput('');
    setIsGenerating(true);

    try {
      // Save user message
      const { data: userMessage, error: userError } = await (supabase
        .from('messages')
        .insert as any)({
          content: userMessageContent,
          role: 'USER',
          chat_id: projectId,
        })
        .select()
        .single();

      if (userError) throw userError;

      setMessages(prev => [...prev, userMessage]);

      // Call AI generation API
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userMessageContent,
          modelKey,
          chatId: projectId,
          userId: user.id,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'ai-stream':
                fullResponse += data.data.content;
                const parsedFiles = parseFilesFromContent(fullResponse);
                if (Object.keys(parsedFiles).length > 0) {
                  setGeneratedFiles(parsedFiles);
                }
                break;

              case 'sandbox':
                // Sandbox created
                break;

              case 'deployment-url':
                setSandboxUrl(data.data.url);
                setActiveTab('sandbox');
                break;

              case 'complete':
                setSandboxUrl(data.data.deploymentUrl);
                
                // Save assistant message
                if (supabase) {
                  await (supabase
                    .from('messages')
                    .insert as any)({
                    content: fullResponse || 'Model training completed successfully!',
                    role: 'ASSISTANT',
                    chat_id: projectId,
                  });

                  await loadMessages();
                }
                break;

              case 'error':
                // Save error message
                if (supabase) {
                  await (supabase
                    .from('messages')
                    .insert as any)({
                    content: data.data.message,
                    role: 'ASSISTANT',
                    chat_id: projectId,
                  });

                  await loadMessages();
                }
                break;
            }
          }
        }
      }
    } catch (error: any) {
      console.error('Error:', error);
      
      // Save error message
      if (supabase) {
        await (supabase
          .from('messages')
          .insert as any)({
          content: `Error: ${error.message}`,
          role: 'ASSISTANT',
          chat_id: projectId,
        });

        await loadMessages();
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const handleNewChat = async () => {
    if (!user || !supabase) return;

    const { data: newProject, error } = await (supabase
      .from('ai_models')
      .insert as any)({
      name: 'New Model',
      user_id: user.id,
      model_type: 'custom',
      framework: 'pytorch',
    })
      .select()
      .single();

    if (!error && newProject) {
      router.push(`/ai-workspace/${newProject.id}`);
    }
  };

  const handleDownloadCode = () => {
    const entries = Object.entries(generatedFiles);
    if (entries.length === 0) return;
    for (const [path, content] of entries) {
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = path.replace(/[\\/]/g, '_');
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }
  };

  if (loading || !user) {
    return <div className={styles.container}>Loading...</div>;
  }

  return (
    <div className={styles.container}>
      {/* Top bar */}
      <div className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <div className={styles.logoText}>zehanxtech</div>
        </div>
        <div className={styles.topbarRight}>
          <select
            value={modelKey}
            onChange={(e) => setModelKey(e.target.value)}
            className={styles.modelSelect}
            aria-label="Select model"
          >
            {Object.entries(AI_MODELS).map(([key, model]) => (
              <option key={key} value={key}>
                {model.name}
              </option>
            ))}
          </select>
          <button className={styles.secondaryButton} onClick={handleDownloadCode}>
            Download Code
          </button>
          <SignOutButton />
        </div>
      </div>

      {/* Left Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <div className={styles.logo}>
            zehanxtech
          </div>
          <button onClick={handleNewChat} className={styles.newChatButton}>
            + New
          </button>
        </div>
        <div className={styles.chatsList}>
          {projects.map((proj) => (
            <div
              key={proj.id}
              className={`${styles.chatItem} ${proj.id === projectId ? styles.active : ''}`}
              onClick={() => router.push(`/ai-workspace/${proj.id}`)}
            >
              <div className={styles.chatName}>{proj.name}</div>
              <div className={styles.chatDate}>
                {new Date(proj.updatedAt).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className={styles.mainContent}>
        <div className={styles.header}>
          <div className={styles.projectTitle}>{project?.name || 'Untitled Project'}</div>
          <div className={styles.headerActions}>
            <button className={styles.iconButton} onClick={() => router.push('/ai-workspace')}>
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
            </button>
          </div>
        </div>

        <div className={styles.chatArea}>
          <div className={styles.messagesPanel}>
            <div className={styles.messagesContainer}>
              {messages.length === 0 && !isGenerating && (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h2 className={styles.emptyTitle}>What can I help with?</h2>
                  <p className={styles.emptyDescription}>
                    Describe the AI model you want to create
                  </p>
                </div>
              )}

              {messages.map((message) => (
                <div key={message.id} className={styles.message}>
                  <div className={styles.messageHeader}>
                    <div className={styles.avatar}>
                      {message.role === 'USER' ? (user.email?.[0]?.toUpperCase() || 'U') : 'AI'}
                    </div>
                    <div className={styles.messageSender}>
                      {message.role === 'USER' ? 'You' : 'zehanxtech AI'}
                    </div>
                  </div>
                  <div className={styles.messageContent}>{message.content}</div>
                </div>
              ))}

              {isGenerating && (
                <div className={styles.message}>
                  <div className={styles.messageHeader}>
                    <div className={styles.avatar}>AI</div>
                    <div className={styles.messageSender}>zehanxtech AI</div>
                  </div>
                  <div className={styles.messageContent}>
                    <div className={styles.spinner} /> Generating...
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className={styles.inputArea}>
              <form onSubmit={handleSubmit} className={styles.inputForm}>
                <div className={styles.inputWrapper}>
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                    placeholder="Message zehanxtech AI..."
                    className={styles.textarea}
                    disabled={isGenerating}
                    rows={1}
                  />
                </div>
                <button
                  type="submit"
                  disabled={!input.trim() || isGenerating}
                  className={styles.sendButton}
                >
                  {isGenerating ? (
                    <>
                      <div className={styles.spinner} />
                      Generating
                    </>
                  ) : (
                    <>
                      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Send
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Right Panel - Code & Sandbox */}
          <div className={styles.rightPanel}>
            <div className={styles.panelTabs}>
              <button
                className={`${styles.tab} ${activeTab === 'code' ? styles.active : ''}`}
                onClick={() => setActiveTab('code')}
              >
                Code
              </button>
              <button
                className={`${styles.tab} ${activeTab === 'sandbox' ? styles.active : ''}`}
                onClick={() => setActiveTab('sandbox')}
              >
                Sandbox
              </button>
              <button
                className={`${styles.tab} ${activeTab === 'deploy' ? styles.active : ''}`}
                onClick={() => setActiveTab('deploy')}
              >
                Deploy
              </button>
            </div>

            <div className={styles.panelContent}>
              {activeTab === 'code' && (
                <div className={styles.codeView}>
                  {Object.keys(generatedFiles).length === 0 ? (
                    <div className={styles.sandboxPlaceholder}>
                      <p>Generated code will appear here</p>
                    </div>
                  ) : (
                    Object.entries(generatedFiles).map(([path, content]) => (
                      <div key={path} className={styles.fileItem}>
                        <div className={styles.fileName}>{path}</div>
                        <div className={styles.codeBlock}>
                          <pre>{content}</pre>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}

              {activeTab === 'sandbox' && (
                <div className={styles.sandboxView}>
                  {sandboxUrl ? (
                    <iframe src={sandboxUrl} className={styles.sandboxFrame} />
                  ) : (
                    <div className={styles.sandboxPlaceholder}>
                      <p>Sandbox preview will appear here once deployed</p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'deploy' && (
                <div className={styles.deployView}>
                  <DeploymentOptions
                    modelId={projectId}
                    onDeploy={async (type, awsKeys) => {
                      try {
                        let response;
                        if (type === 'aws' && awsKeys) {
                          response = await fetch('/api/deploy/aws', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              modelId: projectId,
                              ...awsKeys,
                            }),
                          });
                        } else if (type === 'e2b') {
                          response = await fetch('/api/deploy/e2b', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ modelId: projectId }),
                          });
                        } else {
                          // Local download
                          const downloadRes = await fetch(`/api/models/${projectId}/download`);
                          const downloadData = await downloadRes.json();
                          if (downloadData.downloadUrl) {
                            window.open(downloadData.downloadUrl, '_blank');
                            alert('Model download started. Check your downloads folder.');
                            return;
                          }
                        }

                        if (response && response.ok) {
                          const data = await response.json();
                          if (data.deploymentUrl) {
                            setSandboxUrl(data.deploymentUrl);
                            setActiveTab('sandbox');
                            alert(`Model deployed successfully! URL: ${data.deploymentUrl}`);
                          }
                        }
                      } catch (error: any) {
                        console.error('Deployment error:', error);
                        alert(`Deployment failed: ${error.message}`);
                      }
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
