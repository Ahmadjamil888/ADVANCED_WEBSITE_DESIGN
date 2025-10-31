import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'

export async function POST(request: NextRequest) {
  try {
    const { eventId, hfToken, userId } = await request.json()

    if (!eventId || !hfToken || !userId) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 })
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

    // For demo purposes, return a mock repo URL immediately
    // In production, you'd wait for the Inngest function to complete
    const repoName = `ai-model-${eventId.split('-').pop()}`
    const repoUrl = `https://huggingface.co/${repoName}`

    return NextResponse.json({ 
      success: true,
      message: 'Model deployed successfully!',
      repoUrl,
      eventId
    })

  } catch (error: any) {
    console.error('Hugging Face deployment error:', error)
    
    return NextResponse.json(
      { error: `Deployment failed: ${error.message}` },
      { status: 500 }
    )
  }
}