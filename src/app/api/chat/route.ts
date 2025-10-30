import { NextRequest, NextResponse } from 'next/server'

const GEMINI_API_KEY = process.env.GEMINI_API_KEY

export async function POST(request: NextRequest) {
  try {
    const { message, model, userId } = await request.json()

    if (!message || !model) {
      return NextResponse.json({ error: 'Message and model are required' }, { status: 400 })
    }

    // Customize the prompt based on the model
    let systemPrompt = ''
    switch (model) {
      case 'assistant':
        systemPrompt = 'You are zehanx AI, a helpful AI assistant created by zehanxtech. Provide helpful, accurate, and friendly responses.'
        break
      case 'quiz':
        systemPrompt = 'You are zehanx AI Quiz Generator. Create educational quizzes, questions, and learning materials. Format your responses clearly with questions and answers.'
        break
      case 'helper':
        systemPrompt = 'You are zehanx AI Helper. Assist users with various tasks, provide step-by-step guidance, and offer practical solutions.'
        break
      case 'image-analyzer':
        systemPrompt = 'You are zehanx AI Image Analyzer. Help users understand and analyze images. If no image is provided, explain what you can do with image analysis.'
        break
      case 'researcher':
        systemPrompt = 'You are zehanx AI Researcher. Provide detailed research, analysis, and insights on topics. Use reliable information and cite sources when possible.'
        break
      case 'doc-maker':
        systemPrompt = 'You are zehanx AI Doc Maker. Help create, format, and structure documents. Provide well-organized content with proper formatting.'
        break
      default:
        systemPrompt = 'You are zehanx AI, a helpful AI assistant created by zehanxtech.'
    }

    const fullPrompt = `${systemPrompt}\n\nUser: ${message}\n\nAssistant:`

    // Call Gemini API
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: fullPrompt
          }]
        }],
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 2048,
        },
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          }
        ]
      }),
    })

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.status}`)
    }

    const data = await response.json()
    
    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
      throw new Error('Invalid response from Gemini API')
    }

    const aiResponse = data.candidates[0].content.parts[0].text

    // Clean up the response to remove any asterisks or unwanted formatting
    const cleanResponse = aiResponse
      .replace(/\*\*/g, '') // Remove bold markdown
      .replace(/\*/g, '') // Remove italic markdown
      .replace(/#{1,6}\s/g, '') // Remove markdown headers
      .trim()

    return NextResponse.json({ 
      response: cleanResponse,
      model: model,
      timestamp: new Date().toISOString()
    })

  } catch (error: any) {
    console.error('Chat API error:', error)
    return NextResponse.json(
      { error: 'Failed to process your request. Please try again.' },
      { status: 500 }
    )
  }
}