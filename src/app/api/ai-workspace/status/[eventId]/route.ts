import { NextRequest, NextResponse } from 'next/server'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params

    if (!eventId) {
      return NextResponse.json({ error: 'Missing eventId' }, { status: 400 })
    }

    // Simulate model completion after a reasonable time
    // In a real implementation, you would check the database or Inngest status
    const modelType = 'text-classification'
    const spaceName = `${modelType}-live-${eventId.split('-').pop()}`
    const spaceUrl = `https://huggingface.co/spaces/dhamia/${spaceName}`
    const apiUrl = `https://api-inference.huggingface.co/models/dhamia/${spaceName}`

    // Simple time-based completion (wait 10 seconds from first call)
    // For testing purposes, we'll use a simple approach
    const isReady = true; // Always ready for immediate testing

    if (isReady) {
      // Return the expected format for the frontend
      return NextResponse.json({
        ready: true,
        model: {
          name: 'Sentiment Analysis Model',
          type: 'text-classification',
          framework: 'pytorch',
          dataset: 'imdb-reviews',
          accuracy: 0.92,
          status: 'completed'
        },
        eventId,
        spaceUrl,
        apiUrl,
        spaceName,
        modelType,
        message: '🟢 Sentiment Analysis model ready for deployment!',
        filesGenerated: ['model.py', 'train.py', 'inference.py', 'app.py', 'requirements.txt', 'README.md'],
        timestamp: new Date().toISOString()
      });
    } else {
      // Still processing
      return NextResponse.json({
        ready: false,
        status: 'processing',
        message: 'Model generation in progress...',
        progress: 50,
        eventId,
        timestamp: new Date().toISOString()
      });
    }

  } catch (error: any) {
    console.error('Status check error:', error)
    return NextResponse.json(
      { error: `Status check failed: ${error.message}` },
      { status: 500 }
    )
  }
}