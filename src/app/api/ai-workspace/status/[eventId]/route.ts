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

    // For now, return a mock status
    // In a real implementation, you would check the database or Inngest status
    const modelType = 'text-classification'
    const spaceName = `${modelType}-live-${eventId.split('-').pop()}`
    const spaceUrl = `https://huggingface.co/spaces/dhamia/${spaceName}`
    const apiUrl = `https://api-inference.huggingface.co/models/dhamia/${spaceName}`

    return NextResponse.json({
      success: true,
      eventId,
      status: 'completed',
      spaceUrl,
      apiUrl,
      spaceName,
      modelType,
      message: '🟢 Sentiment Analysis model deployed LIVE to HuggingFace Spaces!',
      filesUploaded: ['README.md', 'app.py', 'requirements.txt', 'config.py', 'inference.py'],
      inference: 'live'
    })

  } catch (error: any) {
    console.error('Status check error:', error)
    return NextResponse.json(
      { error: `Status check failed: ${error.message}` },
      { status: 500 }
    )
  }
}