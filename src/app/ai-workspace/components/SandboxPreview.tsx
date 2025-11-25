'use client';

import { useEffect, useState } from 'react';
import styles from './SandboxPreview.module.css';

interface SandboxPreviewProps {
  sandboxUrl?: string;
  sandboxId?: string;
}

export function SandboxPreview({ sandboxUrl, sandboxId }: SandboxPreviewProps) {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (sandboxUrl) {
      setIsLoading(true);
      // Give iframe time to load
      const timer = setTimeout(() => setIsLoading(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [sandboxUrl]);

  if (!sandboxUrl) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyContent}>
          <div className={styles.emptyIcon}>
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className={styles.emptyTitle}>No Sandbox Active</h3>
          <p className={styles.emptyDescription}>
            Start training a model to see the live preview here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {isLoading && (
        <div className={styles.loadingOverlay}>
          <div className={styles.loadingContent}>
            <div className={styles.spinner} />
            <p className={styles.loadingText}>Loading sandbox...</p>
          </div>
        </div>
      )}
      
      <div className={styles.content}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <div className={styles.trafficLights}>
              <div className={`${styles.trafficLight} ${styles.trafficLightRed}`} />
              <div className={`${styles.trafficLight} ${styles.trafficLightYellow}`} />
              <div className={`${styles.trafficLight} ${styles.trafficLightGreen}`} />
            </div>
            <div className={styles.urlBar}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
              </svg>
              <span className={styles.url}>{sandboxUrl}</span>
            </div>
          </div>
          <a
            href={sandboxUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.openButton}
          >
            Open in New Tab ↗
          </a>
        </div>
        
        <iframe
          src={sandboxUrl}
          className={styles.iframe}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          onLoad={() => setIsLoading(false)}
        />
      </div>
    </div>
  );
}
