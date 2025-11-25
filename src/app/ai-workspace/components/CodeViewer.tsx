'use client';

import { useState, useEffect } from 'react';
import styles from './CodeViewer.module.css';

interface CodeViewerProps {
  files: Record<string, string>;
  isGenerating?: boolean;
}

export function CodeViewer({ files, isGenerating }: CodeViewerProps) {
  const [selectedFile, setSelectedFile] = useState<string>('');
  const fileNames = Object.keys(files);

  useEffect(() => {
    if (fileNames.length > 0 && !selectedFile) {
      setSelectedFile(fileNames[0]);
    }
  }, [fileNames, selectedFile]);

  if (fileNames.length === 0) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        </div>
        <p className={styles.emptyText}>
          {isGenerating ? 'Generating code...' : 'No code generated yet'}
        </p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* File Tabs */}
      <div className={styles.tabs}>
        {fileNames.map((fileName) => (
          <button
            key={fileName}
            onClick={() => setSelectedFile(fileName)}
            className={`${styles.tab} ${selectedFile === fileName ? styles.tabActive : ''}`}
          >
            <FileIcon fileName={fileName} />
            {fileName}
          </button>
        ))}
      </div>

      {/* Code Content */}
      <div className={styles.codeContainer}>
        <div className={styles.codeHeader}>
          <span className={styles.fileName}>{selectedFile}</span>
          <button
            onClick={() => {
              navigator.clipboard.writeText(files[selectedFile]);
            }}
            className={styles.copyButton}
            title="Copy to clipboard"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
            Copy
          </button>
        </div>
        <pre className={styles.codeBlock}>
          <code className={styles.code}>{files[selectedFile]}</code>
        </pre>
      </div>
    </div>
  );
}

function FileIcon({ fileName }: { fileName: string }) {
  const ext = fileName.split('.').pop();
  
  if (ext === 'py') {
    return (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    );
  }
  
  if (ext === 'json') {
    return (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <path d="M10 12h4M10 16h4" />
      </svg>
    );
  }
  
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}
