'use client';

import { useState } from 'react';
import styles from './ModelCard.module.css';

interface ModelCardProps {
  model: {
    id: string;
    name: string;
    description?: string;
    training_status: string;
    model_type: string;
    framework: string;
    performance_metrics?: any;
    created_at: string;
    deployment_url?: string;
    model_file_path?: string;
    model_file_format?: string;
  };
  onView: (id: string) => void;
  onDelete: (id: string) => void;
  onEdit: (id: string) => void;
  onDownload: (id: string) => void;
}

export function ModelCard({ model, onView, onDelete, onEdit, onDownload }: ModelCardProps) {
  const [showMenu, setShowMenu] = useState(false);
  const accuracy = model.performance_metrics?.accuracy || 0;

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <h3 className={styles.name}>{model.name}</h3>
        <button className={styles.menuButton} onClick={() => setShowMenu(!showMenu)}>
          ⋮
        </button>
        {showMenu && (
          <div className={styles.menu}>
            <button onClick={() => { onView(model.id); setShowMenu(false); }}>View</button>
            <button onClick={() => { onEdit(model.id); setShowMenu(false); }}>Edit</button>
            <button onClick={() => { onDownload(model.id); setShowMenu(false); }}>Download</button>
            <button onClick={() => { onDelete(model.id); setShowMenu(false); }} className={styles.deleteButton}>Delete</button>
          </div>
        )}
      </div>
      {model.description && (
        <p className={styles.description}>{model.description}</p>
      )}
      <div className={styles.meta}>
        <span className={styles.badge}>{model.model_type}</span>
        <span className={styles.badge}>{model.framework}</span>
        <span className={`${styles.status} ${styles[model.training_status]}`}>
          {model.training_status}
        </span>
      </div>
      {model.performance_metrics && (
        <div className={styles.metrics}>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>Accuracy:</span>
            <span className={styles.metricValue}>{(accuracy * 100).toFixed(2)}%</span>
          </div>
        </div>
      )}
      <div className={styles.actions}>
        {model.deployment_url && (
          <a href={model.deployment_url} target="_blank" rel="noreferrer" className={styles.deployButton}>
            View Deployed
          </a>
        )}
        {model.model_file_path && (
          <button onClick={() => onDownload(model.id)} className={styles.downloadButton}>
            Download Model
          </button>
        )}
      </div>
      <div className={styles.footer}>
        Created: {new Date(model.created_at).toLocaleDateString()}
      </div>
    </div>
  );
}

