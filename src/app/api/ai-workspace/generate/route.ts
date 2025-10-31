import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export async function POST(request: NextRequest) {
  try {
    const { chatId, prompt, mode, userId } = await request.json()

    if (!prompt || !mode) {
      return NextResponse.json({ error: 'Prompt and mode are required' }, { status: 400 })
    }

    if (!GEMINI_API_KEY) {
      return NextResponse.json({ error: 'Gemini API key not configured' }, { status: 500 })
    }

    // Initialize Gemini AI
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY)
    const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-pro" })

    // Customize the prompt based on the mode
    let systemPrompt = ''
    switch (mode) {
      case 'models':
        systemPrompt = `You are zehanx AI Model Generator, an expert AI system that creates, trains, and deploys custom AI models. 

Your capabilities:
1. Generate complete PyTorch/TensorFlow model code
2. Find and configure datasets from Kaggle/Hugging Face
3. Create training scripts with proper hyperparameters
4. Generate requirements.txt and setup files
5. Provide deployment instructions for Hugging Face

When a user requests an AI model:
1. Analyze their requirements
2. Choose the best architecture (CNN, LSTM, Transformer, etc.)
3. Find suitable datasets
4. Generate complete, runnable code
5. Provide step-by-step training instructions

Always provide:
- Complete Python code files
- Dataset recommendations with sources
- Training configuration
- Performance metrics to track
- Deployment steps

Format your response with clear sections and code blocks.`
        break
      case 'code':
        systemPrompt = 'You are zehanx AI Code Assistant. Help with code generation, debugging, and optimization. Provide clean, well-commented code with explanations.'
        break
      case 'research':
        systemPrompt = 'You are zehanx AI Researcher. Provide detailed research, analysis, and insights. Use reliable sources and provide citations when possible.'
        break
      case 'app-builder':
        systemPrompt = 'You are zehanx AI App Builder. Help create web applications, APIs, and software solutions. Provide complete project structures and deployment guides.'
        break
      case 'translate':
        systemPrompt = 'You are zehanx AI Translator. Provide accurate translations and language assistance. Support multiple languages and cultural context.'
        break
      case 'fine-tune':
        systemPrompt = 'You are zehanx AI Fine-tuning Expert. Guide users through model fine-tuning processes, dataset preparation, and training optimization.'
        break
      default:
        systemPrompt = 'You are zehanx AI, a helpful AI assistant. Provide accurate, helpful responses while being conversational and friendly.'
    }

    const fullPrompt = `${systemPrompt}

User Request: ${prompt}

Please provide a comprehensive response. If this is about creating an AI model, include:
1. Model architecture recommendation
2. Dataset suggestions with sources
3. Complete code implementation
4. Training instructions
5. Deployment steps

Respond in a clear, structured format with code blocks where appropriate.`

    // Generate content using Gemini
    const result = await geminiModel.generateContent(fullPrompt)
    const response = await result.response
    const aiResponse = response.text()

    // Calculate approximate token usage
    const promptTokens = fullPrompt.length / 4 // Rough estimate
    const completionTokens = aiResponse.length / 4
    const totalTokens = promptTokens + completionTokens

    return NextResponse.json({ 
      response: aiResponse,
      model_used: 'gemini-1.5-pro',
      tokens_used: Math.round(totalTokens),
      mode: mode,
      timestamp: new Date().toISOString()
    })

  } catch (error: any) {
    console.error('AI Workspace API error:', error)
    
    // More specific error messages
    if (error.message?.includes('API key')) {
      return NextResponse.json(
        { error: 'API configuration error. Please contact support.' },
        { status: 500 }
      )
    }
    
    if (error.message?.includes('quota') || error.message?.includes('limit')) {
      return NextResponse.json(
        { error: 'Service temporarily unavailable. Please try again later.' },
        { status: 429 }
      )
    }
    
    if (error.message?.includes('404') || error.message?.includes('not found')) {
      return NextResponse.json(
        { error: 'AI model temporarily unavailable. Please try again later.' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: `AI service error: ${error.message || 'Unknown error'}` },
      { status: 500 }
    )
  }
}