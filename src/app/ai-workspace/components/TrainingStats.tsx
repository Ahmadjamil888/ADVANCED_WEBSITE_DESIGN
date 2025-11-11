'use client';

import { useEffect, useState } from 'react';
import styles from './TrainingStats.module.css';

interface TrainingStatsProps {
  trainingJobId: string;
  onComplete?: () => void;
}

export function TrainingStats({ trainingJobId, onComplete }: TrainingStatsProps) {
  const [stats, setStats] = useState({
    currentEpoch: 0,
    totalEpochs: 10,
    loss: 0,
    accuracy: 0,
    validationLoss: 0,
    validationAccuracy: 0,
    status: 'running',
  });

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/training-jobs/${trainingJobId}/stats`);
        if (response.ok) {
          const data = await response.json();
          setStats(data);
          if (data.status === 'completed' && onComplete) {
            onComplete();
          }
        }
      } catch (error) {
        console.error('Error fetching training stats:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingJobId, onComplete]);

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Training Progress</h3>
      <div className={styles.statsGrid}>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Epoch</div>
          <div className={styles.statValue}>{stats.currentEpoch} / {stats.totalEpochs}</div>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${(stats.currentEpoch / stats.totalEpochs) * 100}%` }}
            />
          </div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Loss</div>
          <div className={styles.statValue}>{stats.loss.toFixed(4)}</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Accuracy</div>
          <div className={styles.statValue}>{(stats.accuracy * 100).toFixed(2)}%</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Val Loss</div>
          <div className={styles.statValue}>{stats.validationLoss.toFixed(4)}</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Val Accuracy</div>
          <div className={styles.statValue}>{(stats.validationAccuracy * 100).toFixed(2)}%</div>
        </div>
      </div>
      <div className={styles.status}>
        Status: <span className={styles.statusValue}>{stats.status}</span>
      </div>
    </div>
  );
}

