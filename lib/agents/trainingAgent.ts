 // Training Agent - Handles training pipeline and GPU resource allocation
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { supabase } from '../db/client';

export class TrainingAgent {
  private llm: ChatOpenAI;

  constructor() {
    this.llm = new ChatOpenAI({
      modelName: 'gpt-4',
      temperature: 0.1,
    });
  }

  // Initialize training run
  async initializeTraining(modelId: string, datasetId: string) {
    try {
      // Get model and dataset info
      const { data: model } = await supabase
        .from('models')
        .select('*')
        .eq('id', modelId)
        .single();

      const { data: dataset } = await supabase
        .from('datasets')
        .select('*')
        .eq('id', datasetId)
        .single();

      // Generate training configuration
      const trainingConfig = await this.generateTrainingConfig(model, dataset);

      // Create training run record
      const { data: trainingRun, error } = await supabase
        .from('training_runs')
        .insert({
          user_id: model.user_id,
          model_id: modelId,
          dataset_id: datasetId,
          status: 'pending',
          batch_size: trainingConfig.batchSize,
          learning_rate: trainingConfig.learningRate,
          loss_function: trainingConfig.lossFunction,
          optimizer: trainingConfig.optimizer,
          total_epochs: trainingConfig.epochs,
          metrics: trainingConfig,
        })
        .select()
        .single();

      if (error) throw error;

      return trainingRun;
    } catch (error) {
      console.error('Training Initialization Error:', error);
      throw error;
    }
  }

  // Start training execution
  async startTraining(trainingRunId: string) {
    try {
      // Update status to initializing
      await supabase
        .from('training_runs')
        .update({
          status: 'initializing',
        })
        .eq('id', trainingRunId);

      // Allocate GPU resources
      const gpuAllocation = await this.allocateGPU(trainingRunId);

      // Update status to running
      await supabase
        .from('training_runs')
        .update({
          status: 'running',
          gpu_used: true,
          gpu_memory_used: gpuAllocation.memoryUsed,
        })
        .eq('id', trainingRunId);

      // Start training loop
      const trainingResult = await this.runTrainingLoop(trainingRunId, gpuAllocation);

      return trainingResult;
    } catch (error) {
      console.error('Training Start Error:', error);

      // Update status to failed
      await supabase
        .from('training_runs')
        .update({
          status: 'failed',
          error_message: error.message,
        })
        .eq('id', trainingRunId);

      throw error;
    }
  }

  // Run training loop
  private async runTrainingLoop(trainingRunId: string, gpuAllocation: any) {
    try {
      const { data: trainingRun } = await supabase
        .from('training_runs')
        .select('*')
        .eq('id', trainingRunId)
        .single();

      const totalEpochs = trainingRun.total_epochs || 10;

      for (let epoch = 1; epoch <= totalEpochs; epoch++) {
        // Simulate training step
        const metrics = await this.simulateTrainingStep(epoch, totalEpochs);

        // Update progress
        const progress = Math.round((epoch / totalEpochs) * 100);

        await supabase
          .from('training_runs')
          .update({
            current_epoch: epoch,
            progress,
          })
          .eq('id', trainingRunId);

        // Log metrics
        await this.logMetrics(trainingRunId, epoch, metrics);

        // Check for early stopping (simulated)
        if (metrics.validationAccuracy > 0.95) {
          break;
        }

        // Simulate training time
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Mark training as completed
      await supabase
        .from('training_runs')
        .update({
          status: 'completed',
          training_time_seconds: totalEpochs * 60, // Mock time
        })
        .eq('id', trainingRunId);

      return {
        status: 'completed',
        finalEpoch: totalEpochs,
        gpuAllocation,
      };
    } catch (error) {
      console.error('Training Loop Error:', error);
      throw error;
    }
  }

  // Generate training configuration
  private async generateTrainingConfig(model: any, dataset: any) {
    const configPrompt = `Generate optimal training configuration for this model and dataset:
    Model: ${JSON.stringify(model)}
    Dataset: ${JSON.stringify(dataset)}

    Consider:
    - Model architecture and complexity
    - Dataset size and characteristics
    - Task type (classification, regression, etc.)
    - Available compute resources
    - Training stability and convergence

    Return JSON with:
    - batchSize
    - learningRate
    - epochs
    - optimizer
    - lossFunction
    - regularization
    - dataAugmentation
    - earlyStopping`;

    const response = await this.llm.invoke([
      new SystemMessage('You are an expert in training configuration and hyperparameter optimization.'),
      new HumanMessage(configPrompt),
    ]);

    return JSON.parse(response.content as string);
  }

  // Allocate GPU resources (simulated)
  private async allocateGPU(trainingRunId: string) {
    // In a real implementation, this would:
    // 1. Check available GPUs
    // 2. Allocate based on requirements
    // 3. Monitor usage

    const allocation = {
      gpuId: 'gpu-0',
      memoryUsed: 8 * 1024 * 1024 * 1024, // 8GB
      utilization: 85,
    };

    // Simulate allocation time
    await new Promise(resolve => setTimeout(resolve, 2000));

    return allocation;
  }

  // Simulate training step (for demo purposes)
  private async simulateTrainingStep(epoch: number, totalEpochs: number) {
    // Simulate realistic training metrics
    const baseLoss = 2.0 - (epoch / totalEpochs) * 1.5; // Decreasing loss
    const noise = (Math.random() - 0.5) * 0.2; // Add some noise
    const loss = Math.max(0.1, baseLoss + noise);

    const baseAccuracy = 0.5 + (epoch / totalEpochs) * 0.4; // Increasing accuracy
    const accuracyNoise = (Math.random() - 0.5) * 0.05;
    const accuracy = Math.min(0.95, baseAccuracy + accuracyNoise);

    const validationAccuracy = accuracy * 0.9 + Math.random() * 0.1; // Slightly lower validation

    return {
      loss,
      accuracy,
      validationAccuracy,
      learningRate: 0.001 * Math.pow(0.95, epoch), // Decaying LR
    };
  }

  // Log training metrics
  private async logMetrics(trainingRunId: string, epoch: number, metrics: any) {
    try {
      const metricsToLog = [
        {
          training_run_id: trainingRunId,
          metric_name: 'loss',
          metric_value: metrics.loss,
          epoch,
        },
        {
          training_run_id: trainingRunId,
          metric_name: 'accuracy',
          metric_value: metrics.accuracy,
          epoch,
        },
        {
          training_run_id: trainingRunId,
          metric_name: 'validation_accuracy',
          metric_value: metrics.validationAccuracy,
          epoch,
        },
        {
          training_run_id: trainingRunId,
          metric_name: 'learning_rate',
          metric_value: metrics.learningRate,
          epoch,
        },
      ];

      await supabase
        .from('metrics')
        .insert(metricsToLog);

    } catch (error) {
      console.error('Metrics Logging Error:', error);
      // Don't throw error for logging failures
    }
  }

  // Get training status
  async getTrainingStatus(trainingRunId: string) {
    try {
      const { data: trainingRun } = await supabase
        .from('training_runs')
        .select('*')
        .eq('id', trainingRunId)
        .single();

      const { data: metrics } = await supabase
        .from('metrics')
        .select('*')
        .eq('training_run_id', trainingRunId)
        .order('epoch', { ascending: false })
        .limit(10);

      return {
        trainingRun,
        recentMetrics: metrics,
      };
    } catch (error) {
      console.error('Get Training Status Error:', error);
      throw error;
    }
  }

  // Stop training
  async stopTraining(trainingRunId: string) {
    try {
      await supabase
        .from('training_runs')
        .update({
          status: 'failed',
          error_message: 'Training stopped by user',
        })
        .eq('id', trainingRunId);

      return { success: true };
    } catch (error) {
      console.error('Stop Training Error:', error);
      throw error;
    }
  }

  // Resume training
  async resumeTraining(trainingRunId: string) {
    try {
      const { data: trainingRun } = await supabase
        .from('training_runs')
        .select('*')
        .eq('id', trainingRunId)
        .single();

      if (trainingRun.status !== 'failed') {
        throw new Error('Can only resume failed training runs');
      }

      // Reset error message and restart
      await supabase
        .from('training_runs')
        .update({
          status: 'running',
          error_message: null,
        })
        .eq('id', trainingRunId);

      // Continue training from last epoch
      const gpuAllocation = await this.allocateGPU(trainingRunId);
      const result = await this.runTrainingLoop(trainingRunId, gpuAllocation);

      return result;
    } catch (error) {
      console.error('Resume Training Error:', error);
      throw error;
    }
  }

  // Get training history
  async getTrainingHistory(modelId: string) {
    try {
      const { data: trainingRuns } = await supabase
        .from('training_runs')
        .select('*')
        .eq('model_id', modelId)
        .order('created_at', { ascending: false });

      const history = await Promise.all(
        trainingRuns.map(async (run) => {
          const { data: metrics } = await supabase
            .from('metrics')
            .select('*')
            .eq('training_run_id', run.id)
            .order('epoch');

          return {
            ...run,
            metrics,
          };
        })
      );

      return history;
    } catch (error) {
      console.error('Get Training History Error:', error);
      throw error;
    }
  }

  // Validate training configuration
  async validateTrainingConfig(config: any) {
    try {
      const validationPrompt = `Validate this training configuration:
      Config: ${JSON.stringify(config)}

      Check for:
      - Reasonable hyperparameter values
      - Compatibility with model architecture
      - Resource requirements feasibility
      - Training stability considerations

      Return validation results with issues and recommendations.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are an expert in training configuration validation.'),
        new HumanMessage(validationPrompt),
      ]);

      return JSON.parse(response.content as string);
    } catch (error) {
      console.error('Config Validation Error:', error);
      throw error;
    }
  }
}

export const trainingAgent = new TrainingAgent();
