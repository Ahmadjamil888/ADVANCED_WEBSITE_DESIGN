'use client';

import styles from './StatusIndicator.module.css';

interface StatusIndicatorProps {
  message: string;
  step?: number;
  total?: number;
  type?: 'info' | 'success' | 'error' | 'warning';
}

export function StatusIndicator({ message, step, total, type = 'info' }: StatusIndicatorProps) {
  const iconStyles = {
    info: styles.iconInfo,
    success: styles.iconSuccess,
    error: styles.iconError,
    warning: styles.iconWarning,
  };

  const icons = {
    info: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    success: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    warning: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  };

  return (
    <div className={styles.container}>
      <div className={`${styles.icon} ${iconStyles[type]}`}>
        {icons[type]}
      </div>
      <div className={styles.content}>
        <div className={styles.bubble}>
          <div className={styles.header}>
            <p className={styles.message}>{message}</p>
            {step && total && (
              <span className={styles.stepCounter}>
                {step}/{total}
              </span>
            )}
          </div>
          {step && total && (
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${(step / total) * 100}%` }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
