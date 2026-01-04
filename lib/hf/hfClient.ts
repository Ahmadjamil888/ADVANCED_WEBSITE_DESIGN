// Hugging Face API Client
import { HfInference } from '@huggingface/inference';
import { config } from '../config';

export const hfInference = new HfInference(config.huggingface.token);

// Helper functions for common HF operations
export class HFClient {
  private client: HfInference;

  constructor() {
    this.client = hfInference;
  }

  // Text generation
  async generateText(prompt: string, model = 'gpt2') {
    try {
      const response = await this.client.textGeneration({
        model,
        inputs: prompt,
        parameters: {
          max_new_tokens: 100,
          temperature: 0.7,
        },
      });
      return response.generated_text;
    } catch (error) {
      console.error('HF Text Generation Error:', error);
      throw error;
    }
  }

  // Text classification
  async classifyText(text: string, model = 'cardiffnlp/twitter-roberta-base-sentiment') {
    try {
      const response = await this.client.textClassification({
        model,
        inputs: text,
      });
      return response;
    } catch (error) {
      console.error('HF Text Classification Error:', error);
      throw error;
    }
  }

  // Feature extraction (embeddings)
  async getEmbeddings(text: string, model = 'sentence-transformers/all-MiniLM-L6-v2') {
    try {
      const response = await this.client.featureExtraction({
        model,
        inputs: text,
      });
      return response;
    } catch (error) {
      console.error('HF Feature Extraction Error:', error);
      throw error;
    }
  }

  // List models
  async listModels(filter?: string) {
    try {
      const response = await fetch(`https://huggingface.co/api/models${filter ? `?filter=${filter}` : ''}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('HF List Models Error:', error);
      throw error;
    }
  }

  // List datasets
  async listDatasets(filter?: string) {
    try {
      const response = await fetch(`https://huggingface.co/api/datasets${filter ? `?filter=${filter}` : ''}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('HF List Datasets Error:', error);
      throw error;
    }
  }
}

export const hfClient = new HFClient();
