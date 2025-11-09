'use client';

import { useState, useEffect } from 'react';
import styles from './ChatMessage.module.css';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
  files?: string[];
}

export function ChatMessage({ role, content, isStreaming, files }: ChatMessageProps) {
  const [displayedContent, setDisplayedContent] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (isStreaming && currentIndex < content.length) {
      const timer = setTimeout(() => {
        setDisplayedContent(content.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 10);
      return () => clearTimeout(timer);
    } else {
      setDisplayedContent(content);
    }
  }, [content, currentIndex, isStreaming]);

  if (role === 'user') {
    return (
      <div className={styles.userMessage}>
        <div className={styles.userBubble}>
          <p className={styles.userText}>{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.assistantMessage}>
      <div className={styles.avatar}>
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      <div className={styles.assistantContent}>
        <div className={styles.assistantBubble}>
          <div className={styles.assistantText}>
            {displayedContent}
            {isStreaming && currentIndex < content.length && (
              <span className={styles.cursor} />
            )}
          </div>
          
          {files && files.length > 0 && (
            <div className={styles.filesSection}>
              <div className={styles.filesLabel}>Generated Files:</div>
              <div className={styles.filesList}>
                {files.map((file, idx) => (
                  <div key={idx} className={styles.fileTag}>
                    📄 {file}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
