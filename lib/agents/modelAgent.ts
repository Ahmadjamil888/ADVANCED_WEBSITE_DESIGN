// Model Agent - Handles model architecture design, training, and optimization
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { hfClient } from '../hf/hfClient';
import { supabase } from '../db/client';

export class ModelAgent {
  private llm: ChatOpenAI;

  constructor() {
    this.llm = new ChatOpenAI({
      modelName: 'gpt-4',
      temperature: 0.1,
    });
  }

  // Design model architecture based on task and dataset
  async designModel(task: string, datasetInfo: any) {
    try {
      const designPrompt = `Design a neural network architecture for this task:
      Task: ${task}
      Dataset Info: ${JSON.stringify(datasetInfo)}

      Consider:
      - Input/output dimensions
      - Model complexity vs dataset size
      - Training efficiency
      - Performance expectations

      Return a JSON with architecture details, hyperparameters, and reasoning.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are an expert machine learning architect.'),
        new HumanMessage(designPrompt),
      ]);

      const design = JSON.parse(response.content as string);

      // Suggest base models from Hugging Face
      const baseModels = await hfClient.listModels(task);

      return {
        ...design,
        suggestedBaseModels: baseModels.slice(0, 5),
      };
    } catch (error) {
      console.error('Model Design Error:', error);
      throw error;
    }
  }

  // Generate training configuration
  async generateTrainingConfig(modelId: string, datasetId: string) {
    try {
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

      const configPrompt = `Generate optimal training configuration:
      Model: ${JSON.stringify(model)}
      Dataset: ${JSON.stringify(dataset)}

      Include:
      - Learning rate schedule
      - Batch size
      - Optimizer settings
      - Loss function
      - Data augmentation
      - Regularization techniques
      - Early stopping criteria

      Return JSON configuration.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a training optimization expert.'),
        new HumanMessage(configPrompt),
      ]);

      return JSON.parse(response.content as string);
    } catch (error) {
      console.error('Training Config Generation Error:', error);
      throw error;
    }
  }

  // Monitor training progress and suggest adjustments
  async monitorTraining(trainingRunId: string) {
    try {
      const { data: trainingRun } = await supabase
        .from('training_runs')
        .select('*, models(*), datasets(*)')
        .eq('id', trainingRunId)
        .single();

      const { data: metrics } = await supabase
        .from('metrics')
        .select('*')
        .eq('training_run_id', trainingRunId)
        .order('timestamp', { ascending: false })
        .limit(10);

      const monitoringPrompt = `Analyze training progress:
      Run: ${JSON.stringify(trainingRun)}
      Recent Metrics: ${JSON.stringify(metrics)}

      Assess:
      - Learning curve health
      - Convergence status
      - Potential issues
      - Recommended adjustments

      Return analysis and suggestions.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a training monitoring expert.'),
        new HumanMessage(monitoringPrompt),
      ]);

      const analysis = JSON.parse(response.content as string);

      // If adjustments are needed, update training run
      if (analysis.adjustments) {
        await supabase
          .from('training_runs')
          .update(analysis.adjustments)
          .eq('id', trainingRunId);
      }

      return analysis;
    } catch (error) {
      console.error('Training Monitoring Error:', error);
      throw error;
    }
  }

  // Optimize model performance
  async optimizeModel(modelId: string) {
    try {
      const { data: model } = await supabase
        .from('models')
        .select('*')
        .eq('id', modelId)
        .single();

      const { data: trainingRuns } = await supabase
        .from('training_runs')
        .select('*')
        .eq('model_id', modelId)
        .order('created_at', { ascending: false })
        .limit(5);

      const optimizationPrompt = `Optimize this model:
      Model: ${JSON.stringify(model)}
      Training History: ${JSON.stringify(trainingRuns)}

      Suggest:
      - Architecture improvements
      - Hyperparameter tuning
      - Regularization techniques
      - Ensemble methods
      - Quantization/compression

      Return optimization recommendations.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a model optimization expert.'),
        new HumanMessage(optimizationPrompt),
      ]);

      return JSON.parse(response.content as string);
    } catch (error) {
      console.error('Model Optimization Error:', error);
      throw error;
    }
  }

  // Generate model documentation
  async generateDocumentation(modelId: string) {
    try {
      const { data: model } = await supabase
        .from('models')
        .select('*')
        .eq('id', modelId)
        .single();

      const { data: trainingRuns } = await supabase
        .from('training_runs')
        .select('*')
        .eq('model_id', modelId);

      const docPrompt = `Generate comprehensive documentation for this model:
      Model: ${JSON.stringify(model)}
      Training Runs: ${JSON.stringify(trainingRuns)}

      Include:
      - Architecture description
      - Training procedure
      - Performance metrics
      - Usage examples
      - Limitations and biases

      Return markdown documentation.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a technical documentation expert.'),
        new HumanMessage(docPrompt),
      ]);

      return response.content as string;
    } catch (error) {
      console.error('Documentation Generation Error:', error);
      throw error;
    }
  }
}

export const modelAgent = new ModelAgent();
