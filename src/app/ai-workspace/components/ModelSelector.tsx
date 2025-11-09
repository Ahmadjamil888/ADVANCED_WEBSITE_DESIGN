'use client';

import { AI_MODELS } from '@/lib/ai/models';
import { useState } from 'react';
import styles from './ModelSelector.module.css';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelKey: string) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const currentModel = AI_MODELS[selectedModel];

  return (
    <div className={styles.container}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={styles.button}
      >
        <div className={styles.modelInfo}>
          <div className={styles.statusDot} />
          <span className={styles.modelName}>{currentModel.name}</span>
        </div>
        <svg
          className={`${styles.arrow} ${isOpen ? styles.arrowOpen : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className={styles.overlay}
            onClick={() => setIsOpen(false)}
          />
          <div className={styles.dropdown}>
            <div className={styles.dropdownContent}>
              <div className={styles.categoryLabel}>GROQ MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'groq')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`${styles.modelOption} ${
                      selectedModel === key ? styles.modelOptionSelected : ''
                    }`}
                  >
                    <div className={styles.modelOptionName}>{model.name}</div>
                    <div className={styles.modelOptionDescription}>{model.description}</div>
                  </button>
                ))}

              <div className={styles.categoryLabel}>GEMINI MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'gemini')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`${styles.modelOption} ${
                      selectedModel === key ? styles.modelOptionSelected : ''
                    }`}
                  >
                    <div className={styles.modelOptionName}>{model.name}</div>
                    <div className={styles.modelOptionDescription}>{model.description}</div>
                  </button>
                ))}

              <div className={styles.categoryLabel}>DEEPSEEK MODELS</div>
              {Object.entries(AI_MODELS)
                .filter(([, model]) => model.provider === 'deepseek')
                .map(([key, model]) => (
                  <button
                    key={key}
                    onClick={() => {
                      onModelChange(key);
                      setIsOpen(false);
                    }}
                    className={`${styles.modelOption} ${
                      selectedModel === key ? styles.modelOptionSelected : ''
                    }`}
                  >
                    <div className={styles.modelOptionName}>{model.name}</div>
                    <div className={styles.modelOptionDescription}>{model.description}</div>
                  </button>
                ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
