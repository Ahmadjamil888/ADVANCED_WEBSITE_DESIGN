export const SANDBOX_TIMEOUT = 60_000 * 10 * 3; // 30 minutes
export const MODEL_TRAINING_TIMEOUT = 60_000 * 30; // 30 minutes
export const MAX_TRAINING_STEPS = 1000;

export interface ModelConfig {
  modelType: 'classification' | 'regression' | 'generative' | 'custom';
  framework: 'pytorch' | 'tensorflow' | 'huggingface';
  inputShape: number[];
  outputShape: number[];
  epochs: number;
  batchSize: number;
  learningRate: number;
  datasetPath: string;
  validationSplit: number;
}

export interface TrainingProgress {
  step: number;
  loss: number;
  accuracy?: number;
  metrics: Record<string, number>;
  status: 'training' | 'completed' | 'failed';
  message?: string;
}

export interface ModelArtifacts {
  modelPath: string;
  metadata: {
    framework: string;
    inputShape: number[];
    outputShape: number[];
    trainingTime: number;
    metrics: Record<string, number>;
    createdAt: string;
  };
}
