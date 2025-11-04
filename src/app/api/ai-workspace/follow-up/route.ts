import { NextRequest, NextResponse } from 'next/server';
import { inngest } from '../../../../inngest/client';

/**
 * Handle Follow-up Conversations and Code Modifications
 * Allows users to ask questions and request changes to their AI models
 */

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      prompt, 
      eventId, 
      previousModelId, 
      conversationHistory = [],
      currentFiles = {}
    } = body;

    if (!prompt || !eventId) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    console.log(`🔄 Processing follow-up conversation for eventId: ${eventId}`);

    // Trigger the follow-up conversation function
    await inngest.send({
      name: "ai/conversation.followup",
      data: {
        prompt,
        eventId,
        previousModelId,
        conversationHistory,
        currentFiles
      }
    });

    // Generate immediate conversational response
    const immediateResponse = generateImmediateResponse(prompt);

    return NextResponse.json({
      success: true,
      eventId,
      message: immediateResponse,
      isFollowUp: true,
      conversationContinues: true
    });

  } catch (error: any) {
    console.error('Follow-up conversation error:', error);
    return NextResponse.json(
      { error: error.message || 'Follow-up conversation failed' },
      { status: 500 }
    );
  }
}

function generateImmediateResponse(prompt: string): string {
  const lowerPrompt = prompt.toLowerCase();
  
  if (lowerPrompt.includes('change') || lowerPrompt.includes('modify') || lowerPrompt.includes('edit')) {
    return `Got it! I'll modify the code for you. Let me understand exactly what you want to change and update the model accordingly...`;
  } else if (lowerPrompt.includes('explain') || lowerPrompt.includes('how') || lowerPrompt.includes('why')) {
    return `Great question! Let me explain that part of the code and how it works. I'll break it down step by step...`;
  } else if (lowerPrompt.includes('add') || lowerPrompt.includes('include') || lowerPrompt.includes('feature')) {
    return `Excellent idea! I'll add that feature to your model. This will make it even better. Let me implement that for you...`;
  } else if (lowerPrompt.includes('improve') || lowerPrompt.includes('better') || lowerPrompt.includes('optimize')) {
    return `Perfect! I'll optimize the model for better performance. Let me enhance it with those improvements...`;
  } else if (lowerPrompt.includes('error') || lowerPrompt.includes('problem') || lowerPrompt.includes('issue')) {
    return `I see the issue! Let me help you fix that problem. I'll analyze what's going wrong and provide a solution...`;
  } else {
    return `I understand what you're looking for. Let me help you with that and make the necessary adjustments to your model...`;
  }
}

export async function GET() {
  return NextResponse.json({
    message: "Follow-up conversation endpoint ready",
    capabilities: [
      "Code modifications",
      "Feature additions", 
      "Performance optimizations",
      "Detailed explanations",
      "Error troubleshooting",
      "Architecture changes"
    ]
  });
}