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
    status: 'queued',
    deployment_url: null as string | null,
    error_message: null as string | null,
  });

  useEffect(() => {
    // Fetch immediately on mount
    const fetchStats = async () => {
      try {
        const response = await fetch(`/api/training-jobs/${trainingJobId}/stats`);
        if (response.ok) {
          const data = await response.json();
          console.log('📊 Training stats updated:', data);
          setStats(prev => ({
            ...prev,
            currentEpoch: data.currentEpoch || 0,
            totalEpochs: data.totalEpochs || 10,
            loss: data.loss || 0,
            accuracy: data.accuracy || 0,
            validationLoss: data.validationLoss || 0,
            validationAccuracy: data.validationAccuracy || 0,
            status: data.status || 'unknown',
            deployment_url: data.deployment_url || null,
            error_message: data.error_message || null,
          }));
          
          if (data.status === 'completed' && data.deployment_url && onComplete) {
            console.log('✅ Training completed! Deployment URL:', data.deployment_url);
            onComplete();
          }
        } else {
          console.error('Failed to fetch stats:', response.status);
        }
      } catch (error) {
        console.error('Error fetching training stats:', error);
      }
    };

    // Fetch immediately
    fetchStats();

    // Then poll every 2 seconds
    const interval = setInterval(fetchStats, 2000);

    return () => clearInterval(interval);
  }, [trainingJobId, onComplete]);

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Training Progress</h3>
      
      {/* Error message if training failed */}
      {stats.error_message && (
        <div style={{ 
          padding: '10px', 
          marginBottom: '10px', 
          backgroundColor: '#ff4444', 
          color: 'white', 
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          ❌ Error: {stats.error_message}
        </div>
      )}
      
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
          <div className={styles.statValue}>{typeof stats.loss === 'number' ? stats.loss.toFixed(4) : '0.0000'}</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Accuracy</div>
          <div className={styles.statValue}>{typeof stats.accuracy === 'number' ? (stats.accuracy * 100).toFixed(2) : '0.00'}%</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Val Loss</div>
          <div className={styles.statValue}>{typeof stats.validationLoss === 'number' ? stats.validationLoss.toFixed(4) : '0.0000'}</div>
        </div>
        <div className={styles.statItem}>
          <div className={styles.statLabel}>Val Accuracy</div>
          <div className={styles.statValue}>{typeof stats.validationAccuracy === 'number' ? (stats.validationAccuracy * 100).toFixed(2) : '0.00'}%</div>
        </div>
      </div>
      
      <div className={styles.status}>
        Status: <span className={styles.statusValue}>{stats.status}</span>
      </div>
      
      {/* Show deployment URL when ready */}
      {stats.deployment_url && (
        <div style={{ 
          padding: '10px', 
          marginTop: '10px', 
          backgroundColor: '#44ff44', 
          color: '#000', 
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          ✅ Deployed at: <a href={stats.deployment_url} target="_blank" rel="noreferrer" style={{ color: '#0066cc', textDecoration: 'underline' }}>
            {stats.deployment_url}
          </a>
        </div>
      )}
    </div>
  );
}

