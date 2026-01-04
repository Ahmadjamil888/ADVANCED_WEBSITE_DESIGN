// Dataset Agent - Handles dataset discovery, validation, and preprocessing
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { supabase } from '../db/client';
import { hfClient } from '../hf/hfClient';

export class DatasetAgent {
  private llm: ChatOpenAI;

  constructor() {
    this.llm = new ChatOpenAI({
      modelName: 'gpt-4',
      temperature: 0.1,
    });
  }

  // Discover and recommend datasets for a task
  async discoverDatasets(task: string, requirements: any = {}) {
    try {
      const discoveryPrompt = `Discover and recommend datasets for this ML task:
      Task: ${task}
      Requirements: ${JSON.stringify(requirements)}

      Consider:
      - Task type (classification, regression, generation, etc.)
      - Data format requirements
      - Size constraints
      - Quality and availability
      - Domain relevance

      Search Hugging Face, Kaggle, and other sources.
      Return JSON with recommended datasets and selection rationale.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are an expert data curator and ML dataset specialist.'),
        new HumanMessage(discoveryPrompt),
      ]);

      const recommendations = JSON.parse(response.content as string);

      // Search actual datasets from Hugging Face
      const hfDatasets = await hfClient.listDatasets(task);

      // Filter and rank datasets
      const filteredDatasets = this.filterDatasets(hfDatasets, recommendations.criteria);

      return {
        recommendations,
        availableDatasets: filteredDatasets.slice(0, 10),
        selectedDataset: filteredDatasets[0], // Auto-select best match
      };
    } catch (error) {
      console.error('Dataset Discovery Error:', error);
      throw error;
    }
  }

  // Validate dataset quality and suitability
  async validateDataset(datasetId: string) {
    try {
      // Get dataset info from database
      const { data: dataset, error } = await supabase
        .from('datasets')
        .select('*')
        .eq('id', datasetId)
        .single();

      if (error) throw error;

      const validationPrompt = `Validate this dataset for ML training:
      Dataset: ${JSON.stringify(dataset)}

      Check:
      - Data quality and completeness
      - Format consistency
      - Size adequacy
      - Label accuracy (if applicable)
      - Bias and fairness issues
      - Privacy concerns

      Return JSON validation report with issues and recommendations.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a data validation expert.'),
        new HumanMessage(validationPrompt),
      ]);

      const validation = JSON.parse(response.content as string);

      // Update dataset validation status
      await supabase
        .from('datasets')
        .update({
          validation_status: validation.isValid ? 'valid' : 'invalid',
          validation_errors: validation.errors,
        })
        .eq('id', datasetId);

      return validation;
    } catch (error) {
      console.error('Dataset Validation Error:', error);
      throw error;
    }
  }

  // Preprocess dataset for training
  async preprocessDataset(datasetId: string) {
    try {
      const { data: dataset, error } = await supabase
        .from('datasets')
        .select('*')
        .eq('id', datasetId)
        .single();

      if (error) throw error;

      const preprocessPrompt = `Design preprocessing pipeline for this dataset:
      Dataset: ${JSON.stringify(dataset)}

      Include:
      - Data cleaning steps
      - Feature engineering
      - Normalization/standardization
      - Train/validation/test splits
      - Data augmentation (if applicable)

      Return JSON preprocessing configuration.`;

      const response = await this.llm.invoke([
        new SystemMessage('You are a data preprocessing expert.'),
        new HumanMessage(preprocessPrompt),
      ]);

      const preprocessing = JSON.parse(response.content as string);

      // Update dataset preprocessing status
      await supabase
        .from('datasets')
        .update({
          preprocessing_status: 'processing',
        })
        .eq('id', datasetId);

      // Execute preprocessing (simulated)
      const processedPath = await this.executePreprocessing(dataset, preprocessing);

      // Update dataset with processed path
      await supabase
        .from('datasets')
        .update({
          preprocessing_status: 'completed',
          processed_path: processedPath,
        })
        .eq('id', datasetId);

      return {
        preprocessing,
        processedPath,
      };
    } catch (error) {
      console.error('Dataset Preprocessing Error:', error);

      // Update status to failed
      await supabase
        .from('datasets')
        .update({
          preprocessing_status: 'failed',
        })
        .eq('id', datasetId);

      throw error;
    }
  }

  // Filter datasets based on criteria
  private filterDatasets(datasets: any[], criteria: any) {
    return datasets.filter(dataset => {
      // Basic filtering logic
      const hasRequiredFormat = !criteria.format || dataset.cardData?.format === criteria.format;
      const hasMinSize = !criteria.minSize || (dataset.downloads || 0) >= criteria.minSize;
      const isRelevant = dataset.description?.toLowerCase().includes(criteria.task?.toLowerCase());

      return hasRequiredFormat && hasMinSize && isRelevant;
    });
  }

  // Execute preprocessing (simulated)
  private async executePreprocessing(dataset: any, config: any): Promise<string> {
    // In a real implementation, this would:
    // 1. Download the dataset
    // 2. Apply preprocessing steps
    // 3. Save processed data
    // 4. Return the path

    // For simulation, return mock path
    const processedPath = `/processed/${dataset.id}_${Date.now()}`;

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 3000));

    return processedPath;
  }

  // Get dataset statistics
  async getDatasetStats(datasetId: string) {
    try {
      const { data: dataset, error } = await supabase
        .from('datasets')
        .select('*')
        .eq('id', datasetId)
        .single();

      if (error) throw error;

      // Calculate statistics
      const stats = {
        totalSamples: dataset.num_samples || 0,
        totalFeatures: dataset.num_features || 0,
        dataTypes: dataset.data_types || {},
        sizeBytes: dataset.size_bytes || 0,
        validationStatus: dataset.validation_status,
        preprocessingStatus: dataset.preprocessing_status,
      };

      return stats;
    } catch (error) {
      console.error('Get Dataset Stats Error:', error);
      throw error;
    }
  }

  // Import dataset from external source
  async importDataset(source: string, sourceUrl: string, userId: string) {
    try {
      // Create dataset record
      const { data: dataset, error } = await supabase
        .from('datasets')
        .insert({
          user_id: userId,
          name: `Imported from ${source}`,
          source,
          source_url: sourceUrl,
          validation_status: 'pending',
          preprocessing_status: 'pending',
        })
        .select()
        .single();

      if (error) throw error;

      // Start validation process
      const validation = await this.validateDataset(dataset.id);

      return {
        dataset,
        validation,
      };
    } catch (error) {
      console.error('Dataset Import Error:', error);
      throw error;
    }
  }
}

export const datasetAgent = new DatasetAgent();
