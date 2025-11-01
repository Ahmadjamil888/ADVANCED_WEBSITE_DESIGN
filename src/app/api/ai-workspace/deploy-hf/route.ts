import { NextRequest, NextResponse } from 'next/server'
import { inngest } from '../../../../inngest/client'

async function getHuggingFaceUsername(hfToken: string): Promise<string> {
  try {
    console.log('Getting HF username with token...');
    const response = await fetch('https://huggingface.co/api/whoami', {
      headers: {
        'Authorization': `Bearer ${hfToken}`
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('HF API response:', data);
      if (data.name) {
        return data.name;
      }
    } else {
      console.error('HF API error:', response.status, await response.text());
    }
  } catch (error) {
    console.error('Failed to get HF username:', error);
  }
  
  // If we can't get the username, throw an error instead of using fallback
  throw new Error('Could not authenticate with HuggingFace token. Please check your token.');
}

export async function POST(request: NextRequest) {
  try {
    const { eventId, userId, prompt } = await request.json()

    if (!eventId || !userId || !prompt) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 })
    }

    // Get HuggingFace token from environment variables
    const hfToken = process.env.HUGGINGFACE_TOKEN
    console.log('🔑 HF Token check:', {
      exists: !!hfToken,
      length: hfToken ? hfToken.length : 0,
      startsWithHf: hfToken ? hfToken.startsWith('hf_') : false
    });
    
    if (!hfToken) {
      return NextResponse.json({ error: 'HuggingFace token not configured' }, { status: 500 })
    }

    // Send event to Inngest for Space deployment
    const result = await inngest.send({
      name: "ai/model.deploy-hf",
      data: {
        eventId,
        userId,
        prompt,
        hfToken // Pass the token directly to Inngest
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