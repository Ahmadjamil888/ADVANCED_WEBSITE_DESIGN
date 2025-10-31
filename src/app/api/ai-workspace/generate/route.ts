import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'
import { supabase } from '@/lib/supabase'

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

    // Return immediate response while Inngest processes in background
    const response = generateImmediateResponse(modelConfig, mode)

    return NextResponse.json({ 
      response,
      model_used: 'zehanx-ai-builder',
      tokens_used: Math.round(prompt.length / 4 + response.length / 4),
      mode: mode,
      eventId,
      status: 'processing',
      timestamp: new Date().toISOString()
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
  let modelType = 'classification'
  if (lowerPrompt.includes('image') || lowerPrompt.includes('vision') || lowerPrompt.includes('cnn')) {
    modelType = 'computer-vision'
  } else if (lowerPrompt.includes('text') || lowerPrompt.includes('nlp') || lowerPrompt.includes('sentiment')) {
    modelType = 'text-classification'
  } else if (lowerPrompt.includes('chat') || lowerPrompt.includes('conversation') || lowerPrompt.includes('llm')) {
    modelType = 'language-model'
  } else if (lowerPrompt.includes('regression') || lowerPrompt.includes('predict')) {
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
  }

  return {
    name: extractModelName(prompt) || `AI Model ${Date.now()}`,
    description: prompt,
    modelType,
    framework,
    baseModel,
    domain: extractDomain(prompt),
    requirements: extractRequirements(prompt)
  }
}

function extractModelName(prompt: string): string | null {
  const namePatterns = [
    /create (?:a |an )?(.+?) model/i,
    /build (?:a |an )?(.+?) model/i,
    /(?:model|system) (?:called|named) (.+)/i,
    /(.+?) (?:classifier|predictor|model)/i
  ]
  
  for (const pattern of namePatterns) {
    const match = prompt.match(pattern)
    if (match && match[1]) {
      return match[1].trim()
    }
  }
  
  return null
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
  const { name, modelType, framework, baseModel } = modelConfig
  
  return `🚀 **AI Model Generation Started**

I'm now creating your **${name}** using the zehanx AI Builder system!

**Model Configuration:**
- **Type:** ${modelType.replace('-', ' ').toUpperCase()}
- **Framework:** ${framework.toUpperCase()}
- **Base Model:** ${baseModel}

**What's happening now:**
1. 🔍 **Analyzing Requirements** - Understanding your specifications
2. 🗃️ **Finding Datasets** - Searching Kaggle and Hugging Face for suitable data
3. 🏗️ **Generating Architecture** - Creating optimized model structure
4. 📝 **Writing Code** - Generating complete training and inference scripts
5. ⚙️ **Setting up Environment** - Preparing E2B sandbox for execution

**Next Steps:**
- I'll provide complete code files in a few moments
- Dataset recommendations with download links
- Training instructions and hyperparameter suggestions
- Deployment guide for Hugging Face Hub

This process typically takes 30-60 seconds. I'm working on it now! 🤖✨

*Powered by zehanx AI Builder - Building AI that builds AI*`
}