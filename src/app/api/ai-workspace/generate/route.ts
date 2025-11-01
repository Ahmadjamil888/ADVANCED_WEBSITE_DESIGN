import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'
import { supabase } from '@/lib/supabase'

async function getHuggingFaceUsername(hfToken: string): Promise<string> {
  try {
    console.log('Getting HF username with token...');
    const response = await fetch('https://huggingface.co/api/whoami', {
      headers: {
        'Authorization': `Bearer ${hfToken}`
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('HF API response:', data);
      if (data.name) {
        return data.name;
      }
    } else {
      console.error('HF API error:', response.status, await response.text());
    }
  } catch (error) {
    console.error('Failed to get HF username:', error);
  }
  
  // If we can't get the username, throw an error instead of using fallback
  throw new Error('Could not authenticate with HuggingFace token. Please check your token.');
}

export async function POST(request: NextRequest) {
  try {
    const { chatId, prompt, mode, userId } = await request.json()

    if (!prompt || !mode || !userId) {
      return NextResponse.json({ error: 'Prompt, mode, and userId are required' }, { status: 400 })
    }

    // Parse the user's request to extract model requirements
    const modelConfig = parseModelRequest(prompt, mode)

    // Send event to Inngest to start AI model generation
    const eventId = `ai-model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    await inngest.send({
      name: "ai/model.generate",
      data: {
        eventId,
        userId,
        chatId,
        prompt,
        mode,
        modelConfig,
        timestamp: new Date().toISOString()
      }
    })

    // Also trigger deployment immediately like sentiment analysis route
    const hfToken = process.env.HUGGINGFACE_TOKEN
    let deploymentData = null;
    
    if (hfToken) {
      try {
        const deployResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/ai-workspace/deploy-hf`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            eventId,
            userId,
            prompt
          })
        });
        
        if (deployResponse.ok) {
          deploymentData = await deployResponse.json();
          console.log('Deployment initiated successfully:', deploymentData);
        } else {
          console.error('Deployment failed:', await deployResponse.text());
        }
      } catch (error) {
        console.error('Auto-deploy error:', error);
      }
    }

    // Return immediate response while Inngest processes in background
    const response = generateImmediateResponse(modelConfig, mode)

    // Get actual username from HF token - no fallbacks
    const username = hfToken ? await getHuggingFaceUsername(hfToken) : null;
    if (!username) {
      return NextResponse.json({ error: 'Could not authenticate with HuggingFace token' }, { status: 500 });
    }
    const spaceName = deploymentData?.spaceName || `${modelConfig.modelType}-live-${eventId.split('-').pop()}`;
    const spaceUrl = deploymentData?.spaceUrl || `https://huggingface.co/spaces/${username}/${spaceName}`;
    const apiUrl = deploymentData?.apiUrl || `https://api-inference.huggingface.co/models/${username}/${spaceName}`;

    return NextResponse.json({ 
      response,
      model_used: 'zehanx-ai-builder',
      tokens_used: Math.round(prompt.length / 4 + response.length / 4),
      mode: mode,
      eventId,
      status: 'processing',
      modelType: modelConfig.modelType,
      spaceUrl,
      apiUrl,
      spaceName,
      deploymentStatus: deploymentData?.success ? 'Deployment initiated successfully!' : 'Building live inference Space...',
      timestamp: new Date().toISOString(),
      deploymentData: deploymentData || {
        spaceUrl,
        apiUrl,
        spaceName,
        modelType: modelConfig.modelType,
        status: 'Building...',
        message: 'Space deployment initiated - will be live in 2-3 minutes'
      }
    })

  } catch (error: any) {
    console.error('AI Workspace API error:', error)
    
    return NextResponse.json(
      { error: `AI service error: ${error.message || 'Unknown error'}` },
      { status: 500 }
    )
  }
}

function parseModelRequest(prompt: string, mode: string) {
  const lowerPrompt = prompt.toLowerCase()
  
  // Detect model type
  let modelType = 'text-classification'
  if (lowerPrompt.includes('image') || lowerPrompt.includes('vision') || lowerPrompt.includes('cnn') || lowerPrompt.includes('computer vision')) {
    modelType = 'computer-vision'
  } else if (lowerPrompt.includes('text') || lowerPrompt.includes('nlp') || lowerPrompt.includes('sentiment') || lowerPrompt.includes('classification')) {
    modelType = 'text-classification'
  } else if (lowerPrompt.includes('chat') || lowerPrompt.includes('conversation') || lowerPrompt.includes('llm') || lowerPrompt.includes('language model')) {
    modelType = 'language-model'
  } else if (lowerPrompt.includes('regression') || lowerPrompt.includes('predict') || lowerPrompt.includes('forecasting')) {
    modelType = 'regression'
  }

  // Detect framework preference
  let framework = 'pytorch'
  if (lowerPrompt.includes('tensorflow') || lowerPrompt.includes('keras')) {
    framework = 'tensorflow'
  }

  // Detect base model
  let baseModel = 'bert-base-uncased'
  if (lowerPrompt.includes('gpt')) {
    baseModel = 'gpt2'
  } else if (lowerPrompt.includes('roberta')) {
    baseModel = 'roberta-base'
  } else if (lowerPrompt.includes('resnet')) {
    baseModel = 'resnet50'
  } else if (lowerPrompt.includes('distilbert')) {
    baseModel = 'distilbert-base-uncased'
  }

  return {
    name: extractModelName(prompt) || generateModelName(modelType, prompt),
    description: prompt,
    modelType,
    framework,
    baseModel,
    domain: extractDomain(prompt),
    requirements: extractRequirements(prompt)
  }
}

function extractModelName(prompt: string): string | null {
  // First, try to detect specific model types from the prompt
  const lowerPrompt = prompt.toLowerCase()
  
  // Direct model type detection
  if (lowerPrompt.includes('sentiment analysis') || (lowerPrompt.includes('sentiment') && lowerPrompt.includes('classification'))) {
    return 'Sentiment Analysis Model'
  }
  if (lowerPrompt.includes('spam detection') || (lowerPrompt.includes('spam') && lowerPrompt.includes('classification'))) {
    return 'Spam Detection Model'
  }
  if (lowerPrompt.includes('emotion classification') || (lowerPrompt.includes('emotion') && lowerPrompt.includes('classification'))) {
    return 'Emotion Classification Model'
  }
  if (lowerPrompt.includes('image classification') || (lowerPrompt.includes('image') && lowerPrompt.includes('classification'))) {
    return 'Image Classification Model'
  }
  if (lowerPrompt.includes('object detection')) {
    return 'Object Detection Model'
  }
  if (lowerPrompt.includes('face recognition')) {
    return 'Face Recognition Model'
  }
  
  // Pattern-based extraction with better filtering
  const namePatterns = [
    /(?:create|build|make|generate|develop) (?:a |an |me )?(.+?) (?:model|classifier|predictor|system)/i,
    /(?:model|system|classifier) (?:for|to) (.+)/i,
    /(.+?) (?:classification|detection|prediction|recognition) (?:model|system)/i,
    /(?:model|system) (?:called|named) (.+)/i
  ]
  
  for (const pattern of namePatterns) {
    const match = prompt.match(pattern)
    if (match && match[1]) {
      let name = match[1].trim()
      
      // Clean up the extracted name
      name = name.replace(/^(a |an |the |me |my )/i, '')
      name = name.replace(/(for|to|that|which)$/i, '')
      
      // Filter out common words and short names
      const excludeWords = ['ai', 'machine learning', 'deep learning', 'neural network', 'classification', 'model', 'system']
      if (!excludeWords.includes(name.toLowerCase()) && name.length > 3) {
        // Capitalize properly
        return name.split(' ').map(word => 
          word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
        ).join(' ') + ' Model'
      }
    }
  }
  
  return null
}

function generateModelName(modelType: string, prompt: string): string {
  const lowerPrompt = prompt.toLowerCase()
  
  if (modelType === 'text-classification') {
    if (lowerPrompt.includes('sentiment')) return 'Sentiment Analysis Model'
    if (lowerPrompt.includes('spam')) return 'Spam Detection Model'
    if (lowerPrompt.includes('emotion')) return 'Emotion Classification Model'
    if (lowerPrompt.includes('topic')) return 'Topic Classification Model'
    return 'Text Classification Model'
  } else if (modelType === 'computer-vision') {
    if (lowerPrompt.includes('face')) return 'Face Recognition Model'
    if (lowerPrompt.includes('object')) return 'Object Detection Model'
    if (lowerPrompt.includes('medical')) return 'Medical Image Analysis Model'
    return 'Image Classification Model'
  } else if (modelType === 'language-model') {
    return 'Custom Language Model'
  } else if (modelType === 'regression') {
    if (lowerPrompt.includes('price')) return 'Price Prediction Model'
    if (lowerPrompt.includes('sales')) return 'Sales Forecasting Model'
    return 'Regression Model'
  }
  
  return 'Custom AI Model'
}

function extractDomain(prompt: string): string {
  const lowerPrompt = prompt.toLowerCase()
  
  if (lowerPrompt.includes('medical') || lowerPrompt.includes('health')) return 'healthcare'
  if (lowerPrompt.includes('finance') || lowerPrompt.includes('trading')) return 'finance'
  if (lowerPrompt.includes('ecommerce') || lowerPrompt.includes('retail')) return 'ecommerce'
  if (lowerPrompt.includes('social') || lowerPrompt.includes('media')) return 'social-media'
  if (lowerPrompt.includes('education') || lowerPrompt.includes('learning')) return 'education'
  
  return 'general'
}

function extractRequirements(prompt: string): string[] {
  const requirements = []
  const lowerPrompt = prompt.toLowerCase()
  
  if (lowerPrompt.includes('real-time') || lowerPrompt.includes('fast')) {
    requirements.push('low-latency')
  }
  if (lowerPrompt.includes('accurate') || lowerPrompt.includes('precision')) {
    requirements.push('high-accuracy')
  }
  if (lowerPrompt.includes('scalable') || lowerPrompt.includes('large')) {
    requirements.push('scalable')
  }
  if (lowerPrompt.includes('deploy') || lowerPrompt.includes('production')) {
    requirements.push('production-ready')
  }
  
  return requirements
}

function generateImmediateResponse(modelConfig: any, mode: string): string {
  const { name, modelType, framework, baseModel, domain } = modelConfig
  
  return `## 🤖 AI Model Generation Initiated

I'm now building your **${name}** using advanced machine learning techniques and industry best practices.

### 📋 Model Specifications
- **Architecture Type:** ${modelType.replace('-', ' ').toUpperCase()}
- **Framework:** ${framework.toUpperCase()}
- **Base Model:** ${baseModel}
- **Domain:** ${domain.charAt(0).toUpperCase() + domain.slice(1)}

### 🔄 Current Process Status
The AI model generation pipeline is now active. Here's what's happening:

**Phase 1: Requirements Analysis** ✅
- Parsed your specifications and requirements
- Determined optimal model architecture
- Selected appropriate base model and framework

**Phase 2: Dataset Curation** 🔄
- Searching Kaggle and Hugging Face repositories
- Evaluating dataset quality and relevance
- Preparing data preprocessing pipelines

**Phase 3: Code Generation** 🔄
- Creating model architecture code
- Generating training and validation scripts
- Setting up inference and deployment code

**Phase 4: Environment Setup** 🔄
- Initializing E2B sandbox environment
- Installing dependencies and requirements
- Configuring training environment

**Phase 5: Final Assembly** ⏳
- Packaging complete model project
- Generating documentation and guides
- Preparing deployment options

### ⏱️ Estimated Completion
Your model will be ready in approximately **45-90 seconds**. I'll provide you with:
- Complete source code files
- Training and inference scripts
- Dataset recommendations and setup
- Deployment options (Hugging Face Hub, local, cloud)
- Performance optimization suggestions

Please keep this chat open while I complete the generation process. You'll receive the complete model package shortly.

---
*zehanx AI Builder - Autonomous AI Development Platform*`
}