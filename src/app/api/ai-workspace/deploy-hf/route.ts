import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'

async function getHuggingFaceUsername(hfToken: string): Promise<string> {
  try {
    const response = await fetch('https://huggingface.co/api/whoami', {
      headers: {
        'Authorization': `Bearer ${hfToken}`
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      return data.name || 'user';
    }
  } catch (error) {
    console.error('Failed to get HF username:', error);
  }
  
  // Fallback username
  return 'zehanxtech';
}

export async function POST(request: NextRequest) {
  try {
    const { eventId, userId, prompt } = await request.json()

    if (!eventId || !userId || !prompt) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 })
    }

    // Get HuggingFace token from environment variables
    const hfToken = process.env.HUGGINGFACE_TOKEN
    if (!hfToken) {
      return NextResponse.json({ error: 'HuggingFace token not configured' }, { status: 500 })
    }

    // Send event to Inngest for Space deployment
    const result = await inngest.send({
      name: "ai/model.deploy-hf",
      data: {
        eventId,
        userId,
        prompt
      }
    });

    // Get actual HuggingFace username and generate predicted Space URL
    const username = await getHuggingFaceUsername(hfToken);
    const modelType = prompt.toLowerCase().includes('sentiment') ? 'text-classification' : 
                     prompt.toLowerCase().includes('image') ? 'image-classification' : 'text-classification';
    const spaceName = `${modelType}-live-${eventId.split('-').pop()}`;
    const predictedSpaceUrl = `https://huggingface.co/spaces/${username}/${spaceName}`;

    return NextResponse.json({
      success: true,
      message: '🟢 HuggingFace Space deployment initiated - Building live inference...',
      eventId,
      inngestEventId: result.ids[0],
      spaceUrl: predictedSpaceUrl,
      apiUrl: `https://api-inference.huggingface.co/models/${username}/${spaceName}`,
      spaceName,
      modelType,
      username,
      status: 'Building live inference Space...',
      note: 'Space will be live in 2-3 minutes'
    });

  } catch (error: any) {
    console.error('Deployment error:', error)
    return NextResponse.json(
      { error: `Deployment failed: ${error.message}` },
      { status: 500 }
    )
  }
}