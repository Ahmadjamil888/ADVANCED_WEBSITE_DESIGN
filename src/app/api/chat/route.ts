import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const { message, model, userId } = await request.json()

    if (!message || !model) {
      return NextResponse.json({ error: 'Message and model are required' }, { status: 400 })
    }

    if (!GEMINI_API_KEY) {
      return NextResponse.json({ error: 'Gemini API key not configured' }, { status: 500 })
    }

    // Initialize Gemini AI
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY)
    const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-pro" })

    // Customize the prompt based on the model
    let systemPrompt = ''
    switch (model) {
      case 'assistant':
        systemPrompt = 'You are zehanx AI, a helpful AI assistant created by zehanxtech. Provide helpful, accurate, and friendly responses without using asterisks or markdown formatting.'
        break
      case 'quiz':
        systemPrompt = 'You are zehanx AI Quiz Generator. Create educational quizzes, questions, and learning materials. Format your responses clearly with questions and answers without using asterisks or markdown formatting.'
        break
      case 'helper':
        systemPrompt = 'You are zehanx AI Helper. Assist users with various tasks, provide step-by-step guidance, and offer practical solutions without using asterisks or markdown formatting.'
        break
      case 'image-analyzer':
        systemPrompt = 'You are zehanx AI Image Analyzer. Help users understand and analyze images. If no image is provided, explain what you can do with image analysis without using asterisks or markdown formatting.'
        break
      case 'researcher':
        systemPrompt = 'You are zehanx AI Researcher. Provide detailed research, analysis, and insights on topics. Use reliable information and cite sources when possible without using asterisks or markdown formatting.'
        break
      case 'doc-maker':
        systemPrompt = 'You are zehanx AI Doc Maker. Help create, format, and structure documents. Provide well-organized content with proper formatting without using asterisks or markdown formatting.'
        break
      default:
        systemPrompt = 'You are zehanx AI, a helpful AI assistant created by zehanxtech without using asterisks or markdown formatting.'
    }

    const fullPrompt = `${systemPrompt}\n\nUser: ${message}\n\nPlease respond in plain text without any asterisks, bold formatting, or markdown. Keep your response clean and readable.`

    // Generate content using Gemini
    const result = await geminiModel.generateContent(fullPrompt)
    const response = await result.response
    const aiResponse = response.text()

    // Clean up the response to remove any asterisks or unwanted formatting
    const cleanResponse = aiResponse
      .replace(/\*\*/g, '') // Remove bold markdown
      .replace(/\*/g, '') // Remove italic markdown
      .replace(/#{1,6}\s/g, '') // Remove markdown headers
      .replace(/`{1,3}/g, '') // Remove code blocks
      .trim()

    return NextResponse.json({ 
      response: cleanResponse,
      model: model,
      timestamp: new Date().toISOString()
    })

  } catch (error: any) {
    console.error('Chat API error:', error)
    
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