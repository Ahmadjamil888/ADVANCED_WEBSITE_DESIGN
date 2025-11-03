import { NextRequest, NextResponse } from 'next/server'

/**
 * AI Model Training Status Endpoint
 * Tracks the progress of the complete AI model pipeline
 */

// In-memory status tracking (in production, use Redis or database)
const trainingStatus = new Map<string, any>()

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params

    if (!eventId) {
      return NextResponse.json({ error: 'Missing eventId' }, { status: 400 })
    }

    // Get status from memory (in production, query database/Redis)
    const status = trainingStatus.get(eventId) || {
      completed: false,
      currentStage: 'Initializing...',
      progress: 0,
      startTime: new Date().toISOString()
    }

    // Simulate progress updates (replace with real Inngest status)
    if (!status.completed) {
      const elapsed = Date.now() - new Date(status.startTime).getTime()
      const stages = [
        { name: 'Analyzing prompt and finding best model...', duration: 10000 },
        { name: 'Searching Kaggle for optimal dataset...', duration: 20000 },
        { name: 'Generating PyTorch code pipeline...', duration: 30000 },
        { name: 'Setting up E2B training environment...', duration: 40000 },
        { name: 'Training model on dataset...', duration: 120000 }, // 2 minutes
        { name: 'Deploying to HuggingFace with Git CLI...', duration: 150000 },
        { name: 'Finalizing deployment...', duration: 160000 }
      ]

      let currentStageIndex = 0
      for (let i = 0; i < stages.length; i++) {
        if (elapsed < stages[i].duration) {
          currentStageIndex = i
          break
        }
      }

      if (elapsed >= 160000) { // 2.5 minutes total
        status.completed = true
        status.currentStage = 'Completed!'
        status.progress = 100
        status.spaceUrl = `https://huggingface.co/spaces/Ahmadjamil888/text-classification-${eventId.split('-').pop()}`
      } else {
        status.currentStage = stages[currentStageIndex].name
        status.progress = Math.min((elapsed / 160000) * 100, 95)
      }

      trainingStatus.set(eventId, status)
    }

    return NextResponse.json(status)

  } catch (error: any) {
    console.error('Status check error:', error)
    return NextResponse.json(
      { error: `Status check failed: ${error.message}` },
      { status: 500 }
    )
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params
    const body = await request.json()

    if (!eventId) {
      return NextResponse.json({ error: 'Missing eventId' }, { status: 400 })
    }

    // Update status (called by Inngest function)
    trainingStatus.set(eventId, {
      ...body,
      lastUpdated: new Date().toISOString()
    })

    return NextResponse.json({ success: true })

  } catch (error: any) {
    console.error('Status update error:', error)
    return NextResponse.json(
      { error: `Status update failed: ${error.message}` },
      { status: 500 }
    )
  }
}