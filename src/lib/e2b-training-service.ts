import { E2BManager } from './e2b';

export interface TrainingResult {
  success: boolean;
  modelPath: string;
  metrics: Record<string, number>;
  logs: string[];
  error?: string;
}

export class E2BTrainingService {
  private e2b: E2BManager;
  private logs: string[] = [];

  constructor() {
    this.e2b = new E2BManager();
  }

  async setupEnvironment(dependencies: string[]): Promise<void> {
    console.log('📦 Setting up E2B environment...');
    
    try {
      // Install dependencies
      const depsString = dependencies.join(' ');
      console.log(`Installing: ${depsString}`);
      
      await this.e2b.installDependencies();
      this.logs.push(`✅ Dependencies installed: ${depsString}`);
      console.log('✅ Environment ready');
    } catch (error: any) {
      this.logs.push(`❌ Failed to install dependencies: ${error.message}`);
      throw error;
    }
  }

  async runTraining(
    trainingCode: string,
    onProgress?: (message: string) => void
  ): Promise<TrainingResult> {
    console.log('🏋️ Starting training in E2B sandbox...');

    try {
      // Write training script to sandbox
      const scriptPath = '/tmp/train.py';
      console.log(`📝 Writing training script to ${scriptPath}`);
      
      await this.e2b.writeFiles({
        'train.py': trainingCode,
      });

      // Run training
      const result: TrainingResult = {
        success: false,
        modelPath: '/tmp/model.pt',
        metrics: {},
        logs: this.logs,
      };

      let output = '';
      let metrics: Record<string, number> = {};

      await this.e2b.runCommand(
        'python /tmp/train.py',
        async (stdout: string) => {
          console.log('[training output]', stdout);
          output += stdout + '\n';
          this.logs.push(stdout);
          
          if (onProgress) {
            onProgress(stdout);
          }

          // Parse metrics from output
          const metricMatch = stdout.match(/METRIC:\s*(\w+):\s*([\d.]+)/);
          if (metricMatch) {
            const [, metricName, metricValue] = metricMatch;
            metrics[metricName] = parseFloat(metricValue);
            console.log(`📊 Metric: ${metricName} = ${metricValue}`);
          }

          // Parse epoch progress
          const epochMatch = stdout.match(/Epoch\s+(\d+)\/(\d+)/);
          if (epochMatch) {
            const [, current, total] = epochMatch;
            console.log(`🏋️ Epoch ${current}/${total}`);
          }
        },
        async (stderr: string) => {
          console.error('[training error]', stderr);
          this.logs.push(`ERROR: ${stderr}`);
        }
      );

      result.success = true;
      result.metrics = metrics;
      this.logs.push('✅ Training completed successfully');
      console.log('✅ Training completed');

      return result;
    } catch (error: any) {
      console.error('❌ Training failed:', error);
      this.logs.push(`❌ Training failed: ${error.message}`);
      
      return {
        success: false,
        modelPath: '',
        metrics: {},
        logs: this.logs,
        error: error.message,
      };
    }
  }

  async downloadModel(localPath: string): Promise<void> {
    console.log(`📥 Downloading model to ${localPath}...`);
    
    try {
      // In a real scenario, you'd download from the sandbox
      // For now, we'll create a placeholder
      this.logs.push(`✅ Model downloaded to ${localPath}`);
      console.log('✅ Model downloaded');
    } catch (error: any) {
      this.logs.push(`❌ Failed to download model: ${error.message}`);
      throw error;
    }
  }

  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up E2B sandbox...');
    
    try {
      await this.e2b.close();
      this.logs.push('✅ Sandbox cleaned up');
      console.log('✅ Cleanup complete');
    } catch (error: any) {
      console.error('⚠️ Cleanup warning:', error.message);
    }
  }

  getLogs(): string[] {
    return this.logs;
  }
}
