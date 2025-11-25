import { AIClient } from './ai/client';

export interface TrainingPlan {
  task: string;
  model: {
    name: string;
    source: string; // huggingface or pytorch
    pretrained: string;
  };
  dataset: {
    name: string;
    source: string; // huggingface, kaggle, or url
    url: string;
    size: string; // small, medium, large
  };
  framework: string; // pytorch, tensorflow
  dependencies: string[];
  trainingCode: string;
  evaluationMetrics: string[];
  estimatedTime: string; // in minutes
}

export async function generateTrainingPlan(task: string): Promise<TrainingPlan> {
  console.log(`🤖 Generating training plan for task: ${task}`);

  const systemPrompt = `You are an expert AI researcher and ML engineer. Your task is to create a complete, executable training plan for a given ML task.

You MUST output ONLY valid JSON (no markdown, no explanations, no code blocks).

For the given task, provide:
1. Best pretrained model for this task
2. Best dataset (from HuggingFace or Kaggle)
3. Complete, runnable Python training code using PyTorch
4. All required dependencies
5. Key evaluation metrics
6. Estimated training time

Output format:
{
  "task": "the task description",
  "model": {
    "name": "model name",
    "source": "huggingface or pytorch",
    "pretrained": "model identifier (e.g., bert-base-uncased)"
  },
  "dataset": {
    "name": "dataset name",
    "source": "huggingface or kaggle or url",
    "url": "direct download URL or huggingface path",
    "size": "small or medium or large"
  },
  "framework": "pytorch or tensorflow",
  "dependencies": ["torch", "transformers", "datasets", "scikit-learn"],
  "trainingCode": "complete Python code as a single string with proper escaping",
  "evaluationMetrics": ["accuracy", "f1-score", "precision", "recall"],
  "estimatedTime": "10-15 minutes"
}`;

  const userPrompt = `Task: ${task}

Create a complete training plan for this task. The code must be:
- Self-contained and runnable
- Use small/medium datasets for quick training
- Include data loading, preprocessing, model training, and evaluation
- Save the final model to /tmp/model.pt
- Print metrics in format: "METRIC: value"`;

  try {
    const aiClient = new AIClient('groq', 'llama-3.3-70b-versatile');
    let content = '';

    for await (const chunk of aiClient.streamCompletion([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ])) {
      if (!chunk.done) {
        content += chunk.content;
      }
    }

    if (!content) {
      throw new Error('Empty response from Groq');
    }

    console.log('📋 Raw Groq response received, parsing...');
    
    // Extract JSON from response (handle markdown code blocks)
    let jsonStr = content;
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      jsonStr = jsonMatch[1];
    }

    const plan = JSON.parse(jsonStr) as TrainingPlan;
    console.log('✅ Training plan generated successfully');
    console.log(`   Model: ${plan.model.pretrained}`);
    console.log(`   Dataset: ${plan.dataset.name}`);
    console.log(`   Estimated time: ${plan.estimatedTime}`);

    return plan;
  } catch (error: any) {
    console.error('❌ Error generating training plan:', error.message);
    throw new Error(`Failed to generate training plan: ${error.message}`);
  }
}

export async function generateDeploymentCode(
  model: string,
  task: string,
  metrics: Record<string, number>
): Promise<string> {
  console.log(`🚀 Generating deployment code for ${model}`);

  const prompt = `Generate a complete FastAPI + Gradio app for deploying a ${task} model.

The app should:
1. Load the model from /tmp/model.pt
2. Accept input text
3. Return predictions
4. Display metrics: ${JSON.stringify(metrics)}
5. Be production-ready

Output ONLY the Python code, no explanations.`;

  try {
    const aiClient = new AIClient('groq', 'llama-3.3-70b-versatile');
    let code = '';

    for await (const chunk of aiClient.streamCompletion([
      { role: 'system', content: 'You are a Python expert. Output only valid Python code.' },
      { role: 'user', content: prompt },
    ])) {
      if (!chunk.done) {
        code += chunk.content;
      }
    }

    if (!code) {
      throw new Error('Empty response from Groq');
    }

    console.log('✅ Deployment code generated');
    return code;
  } catch (error: any) {
    console.error('❌ Error generating deployment code:', error.message);
    throw new Error(`Failed to generate deployment code: ${error.message}`);
  }
}
