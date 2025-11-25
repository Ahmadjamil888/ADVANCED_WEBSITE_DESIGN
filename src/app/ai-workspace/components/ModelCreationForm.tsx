'use client';

import { useState, useRef } from 'react';
import styles from './ModelCreationForm.module.css';

interface ModelCreationFormProps {
  onSubmit: (data: ModelCreationData) => void;
  onClose: () => void;
}

export interface ModelCreationData {
  prompt: string;
  trainingMode: 'from_scratch' | 'fine_tune';
  datasetFile?: File;
  modelFile?: File;
  extraInstructions?: string;
}

export function ModelCreationForm({ onSubmit, onClose }: ModelCreationFormProps) {
  const [prompt, setPrompt] = useState('');
  const [trainingMode, setTrainingMode] = useState<'from_scratch' | 'fine_tune'>('from_scratch');
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [extraInstructions, setExtraInstructions] = useState('');
  const [showDatasetUpload, setShowDatasetUpload] = useState(false);
  const [showModelUpload, setShowModelUpload] = useState(false);
  const datasetInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    if (trainingMode === 'from_scratch' && !datasetFile) {
      alert('Dataset is required when training from scratch');
      return;
    }

    onSubmit({
      prompt,
      trainingMode,
      datasetFile: datasetFile || undefined,
      modelFile: modelFile || undefined,
      extraInstructions: extraInstructions || undefined,
    });
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>Create AI Model</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label>Describe your AI model</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Create a sentiment analysis model using BERT"
              className={styles.textarea}
              rows={3}
              required
            />
          </div>

          <div className={styles.field}>
            <label>Training Mode</label>
            <div className={styles.toggleGroup}>
              <button
                type="button"
                className={`${styles.toggleButton} ${trainingMode === 'from_scratch' ? styles.active : ''}`}
                onClick={() => setTrainingMode('from_scratch')}
              >
                Make from Scratch
              </button>
              <button
                type="button"
                className={`${styles.toggleButton} ${trainingMode === 'fine_tune' ? styles.active : ''}`}
                onClick={() => setTrainingMode('fine_tune')}
              >
                Fine Tune
              </button>
            </div>
            {trainingMode === 'from_scratch' && (
              <p className={styles.hint}>Dataset is required when training from scratch</p>
            )}
          </div>

          <div className={styles.field}>
            <div className={styles.buttonRow}>
              <button
                type="button"
                className={styles.uploadButton}
                onClick={() => {
                  setShowDatasetUpload(true);
                  datasetInputRef.current?.click();
                }}
              >
                {datasetFile ? `✓ ${datasetFile.name}` : 'Use Your Own Data'}
              </button>
              <input
                ref={datasetInputRef}
                type="file"
                accept=".csv,.json,.txt,.parquet"
                onChange={(e) => setDatasetFile(e.target.files?.[0] || null)}
                style={{ display: 'none' }}
              />
              <button
                type="button"
                className={styles.uploadButton}
                onClick={() => {
                  setShowModelUpload(true);
                  modelInputRef.current?.click();
                }}
                disabled={trainingMode === 'from_scratch'}
              >
                {modelFile ? `✓ ${modelFile.name}` : 'Upload Your Own AI Model'}
              </button>
              <input
                ref={modelInputRef}
                type="file"
                accept=".pth,.h5,.pb,.onnx,.safetensors"
                onChange={(e) => setModelFile(e.target.files?.[0] || null)}
                style={{ display: 'none' }}
                disabled={trainingMode === 'from_scratch'}
              />
            </div>
          </div>

          <div className={styles.field}>
            <label>Extra Instructions (Optional)</label>
            <textarea
              value={extraInstructions}
              onChange={(e) => setExtraInstructions(e.target.value)}
              placeholder="Any additional requirements or specifications..."
              className={styles.textarea}
              rows={2}
            />
          </div>

          <div className={styles.actions}>
            <button type="button" className={styles.cancelButton} onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className={styles.submitButton}>
              Trigger Training
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

