import { NextRequest, NextResponse } from 'next/server';
import { inngest } from '@/inngest/client';
import { createClient } from '@supabase/supabase-js';
import { randomUUID } from 'crypto';

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const supabase = createClient(supabaseUrl, supabaseKey);

export async function POST(request: NextRequest) {
  try {
    const { prompt, chatId, userId } = await request.json();

    if (!prompt || !chatId || !userId) {
      return NextResponse.json(
        { error: 'Missing required fields: prompt, chatId, or userId' },
        { status: 400 }
      );
    }

    // Generate a unique event ID for this training job
    const eventId = randomUUID();

    // Simple model configuration based on prompt
    const modelConfig = {
      type: 'text-classification',
      task: 'Sentiment Analysis',
      baseModel: 'distilbert-base-uncased',
      dataset: 'imdb',
      epochs: 3,
      batchSize: 16,
      learningRate: 2e-5,
      prompt: prompt
    };

    // Send event to Inngest to start the training process
    await inngest.send({
      name: 'ai/model.train',
      data: {
        modelConfig,
        eventId,
        userId,
        chatId,
        timestamp: new Date().toISOString()
      },
    });

    // Create a status entry in the database
    const { error } = await supabase
      .from('training_jobs')
      .insert([
        {
          id: eventId,
          user_id: userId,
          chat_id: chatId,
          status: 'pending',
          model_config: modelConfig,
          created_at: new Date().toISOString()
        }
      ]);

    if (error) {
      console.error('Error creating training job:', error);
      return NextResponse.json(
        { error: 'Failed to create training job' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      eventId,
      message: 'Training job started successfully',
      statusUrl: `/api/ai-workspace/status/${eventId}`
    });

  } catch (error) {
    console.error('Error starting training:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export const runtime = 'nodejs';
