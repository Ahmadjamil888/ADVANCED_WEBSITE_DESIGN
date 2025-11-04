import { NextRequest, NextResponse } from 'next/server';
import { inngest } from '../../../inngest/client';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prompt = "test sentiment analysis model" } = body;
    
    const eventId = `test_${Date.now()}`;
    
    // Test the Inngest function
    await inngest.send({
      name: "ai/model.generate",
      data: {
        eventId,
        userId: "test-user",
        chatId: "test-chat",
        prompt,
        modelConfig: {
          type: 'text-classification',
          task: 'Sentiment Analysis',
          baseModel: 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        }
      }
    });

    return NextResponse.json({
      success: true,
      message: "Inngest function triggered successfully",
      eventId,
      statusUrl: `/api/ai-workspace/status/${eventId}`
    });

  } catch (error: any) {
    console.error('Test Inngest error:', error);
    return NextResponse.json(
      { error: error.message || 'Test failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: "Inngest test endpoint ready",
    functions: [
      "zehanx-ai-workspace-generate-model-code",
      "zehanx-ai-workspace-analyze-prompt",
      "zehanx-ai-workspace-find-dataset", 
      "zehanx-ai-workspace-train-model",
      "zehanx-ai-workspace-deploy-hf"
    ]
  });
}