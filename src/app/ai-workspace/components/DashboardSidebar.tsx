'use client';

import { useState } from 'react';
import styles from './DashboardSidebar.module.css';

interface DashboardSidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export function DashboardSidebar({ activeSection, onSectionChange }: DashboardSidebarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const sections = [
    { id: 'models', label: 'Trained Models', icon: '🤖' },
    { id: 'llms', label: 'LLMs', icon: '🧠' },
    { id: 'datasets', label: 'Datasets', icon: '📊' },
    { id: 'in-progress', label: 'In Progress', icon: '⚙️' },
    { id: 'billing', label: 'Billing', icon: '💳' },
  ];

  return (
    <div
      className={`${styles.sidebar} ${isExpanded ? styles.expanded : ''}`}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      <div className={styles.sidebarContent}>
        {sections.map((section) => (
          <button
            key={section.id}
            className={`${styles.sidebarItem} ${activeSection === section.id ? styles.active : ''}`}
            onClick={() => onSectionChange(section.id)}
            title={section.label}
          >
            <span className={styles.icon}>{section.icon}</span>
            {isExpanded && <span className={styles.label}>{section.label}</span>}
          </button>
        ))}
      </div>
      {isExpanded && (
        <button
          className={styles.closeButton}
          onClick={() => setIsExpanded(false)}
          title="Collapse sidebar"
        >
          ✕
        </button>
      )}
    </div>
  );
}

