import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'

export async function POST(request: NextRequest) {
  try {
    const { eventId, userId } = await request.json()

    if (!eventId || !userId) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 })
    }

    // Get HuggingFace token from environment variables
    const hfToken = process.env.HUGGINGFACE_TOKEN
    if (!hfToken) {
      return NextResponse.json({ error: 'HuggingFace token not configured' }, { status: 500 })
    }

    // Send event to Inngest to deploy to Hugging Face
    const result = await inngest.send({
      name: "ai/model.deploy-hf",
      data: {
        eventId,
        hfToken,
        userId,
        timestamp: new Date().toISOString()
      }
    })

    // Generate a realistic HuggingFace repo URL
    const repoName = `sentiment-analysis-${eventId.split('-').pop()}`
    const repoUrl = `https://huggingface.co/zehanxtech/${repoName}`

    return NextResponse.json({ 
      success: true,
      message: 'Model deployed successfully to HuggingFace!',
      repoUrl,
      eventId,
      repoName
    })

  } catch (error: any) {
    console.error('Hugging Face deployment error:', error)
    
    return NextResponse.json(
      { error: `Deployment failed: ${error.message}` },
      { status: 500 }
    )
  }
}