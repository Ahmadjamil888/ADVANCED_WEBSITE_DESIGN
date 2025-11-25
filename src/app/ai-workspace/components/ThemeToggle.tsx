'use client';

import { useState, useEffect } from 'react';
import styles from './ThemeToggle.module.css';

export function ThemeToggle() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    const saved = localStorage.getItem('dashboard-theme') as 'dark' | 'light' | null;
    if (saved) {
      setTheme(saved);
      document.documentElement.setAttribute('data-dashboard-theme', saved);
    } else {
      document.documentElement.setAttribute('data-dashboard-theme', 'dark');
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('dashboard-theme', newTheme);
    document.documentElement.setAttribute('data-dashboard-theme', newTheme);
  };

  return (
    <button className={styles.toggle} onClick={toggleTheme} title="Toggle theme">
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}
