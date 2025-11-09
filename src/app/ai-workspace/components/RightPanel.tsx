'use client';

import { useState } from 'react';
import { CodeViewer } from './CodeViewer';
import { SandboxPreview } from './SandboxPreview';
import styles from './RightPanel.module.css';

interface RightPanelProps {
  files: Record<string, string>;
  sandboxUrl?: string;
  sandboxId?: string;
  isGenerating?: boolean;
}

export function RightPanel({ files, sandboxUrl, sandboxId, isGenerating }: RightPanelProps) {
  const [activeView, setActiveView] = useState<'code' | 'sandbox'>('code');
  const hasFiles = Object.keys(files).length > 0;

  return (
    <div className={styles.container}>
      {/* Toggle Tabs */}
      <div className={styles.tabs}>
        <button
          onClick={() => setActiveView('code')}
          className={`${styles.tab} ${activeView === 'code' ? styles.tabActive : ''}`}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
          Code
          {hasFiles && <span className={styles.badge}>{Object.keys(files).length}</span>}
        </button>
        <button
          onClick={() => setActiveView('sandbox')}
          className={`${styles.tab} ${activeView === 'sandbox' ? styles.tabActive : ''}`}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          Sandbox
          {sandboxUrl && <span className={styles.badgeGreen}>●</span>}
        </button>
      </div>

      {/* Content */}
      <div className={styles.content}>
        {activeView === 'code' ? (
          <CodeViewer files={files} isGenerating={isGenerating} />
        ) : (
          <SandboxPreview sandboxUrl={sandboxUrl} sandboxId={sandboxId} />
        )}
      </div>
    </div>
  );
}
