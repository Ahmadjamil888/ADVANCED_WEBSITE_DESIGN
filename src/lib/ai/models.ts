// AI Model configurations
export type AIProvider = 'groq' | 'gemini' | 'deepseek';

export interface AIModel {
  id: string;
  name: string;
  provider: AIProvider;
  description: string;
  maxTokens: number;
}

export const AI_MODELS: Record<string, AIModel> = {
  // Groq Models
  'llama-3.3-70b': {
    id: 'llama-3.3-70b-versatile',
    name: 'Llama 3.3 70B',
    provider: 'groq',
    description: 'Fast and powerful for code generation',
    maxTokens: 8000,
  },
  'llama-3.1-8b': {
    id: 'llama-3.1-8b-instant',
    name: 'Llama 3.1 8B',
    provider: 'groq',
    description: 'Lightning fast responses',
    maxTokens: 8000,
  },
  
  // Gemini Models
  'gemini-2.0-flash': {
    id: 'gemini-2.0-flash-exp',
    name: 'Gemini 2.0 Flash',
    provider: 'gemini',
    description: 'Google\'s latest multimodal AI',
    maxTokens: 8000,
  },
  'gemini-1.5-pro': {
    id: 'gemini-1.5-pro',
    name: 'Gemini 1.5 Pro',
    provider: 'gemini',
    description: 'Advanced reasoning and code',
    maxTokens: 8000,
  },
  
  // DeepSeek Models
  'deepseek-chat': {
    id: 'deepseek-chat',
    name: 'DeepSeek Chat',
    provider: 'deepseek',
    description: 'Specialized in coding tasks',
    maxTokens: 8000,
  },
  'deepseek-coder': {
    id: 'deepseek-coder',
    name: 'DeepSeek Coder',
    provider: 'deepseek',
    description: 'Optimized for code generation',
    maxTokens: 8000,
  },
};

export const DEFAULT_MODEL = 'llama-3.3-70b';
