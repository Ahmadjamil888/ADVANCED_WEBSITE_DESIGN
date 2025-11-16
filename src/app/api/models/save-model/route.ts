import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const maxDuration = 60;

export async function POST(request: NextRequest) {
  try {
    const {
      userId,
      modelName,
      modelType,
      description,
      deploymentUrl,
      sandboxId,
      modelPath,
      trainingStats,
      prompt,
    } = await request.json();

    if (!userId || !modelName) {
      return NextResponse.json(
        { error: 'userId and modelName are required' },
        { status: 400 }
      );
    }

    // Initialize Supabase client
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json(
        { error: 'Supabase configuration missing' },
        { status: 500 }
      );
    }

    const supabase = createClient(supabaseUrl, supabaseKey);

    // Save model to database
    const { data, error } = await supabase
      .from('ai_models')
      .insert([
        {
          user_id: userId,
          name: modelName,
          type: modelType,
          description: description || '',
          deployment_url: deploymentUrl || null,
          sandbox_id: sandboxId || null,
          model_path: modelPath || null,
          training_stats: trainingStats || {},
          prompt: prompt || '',
          created_at: new Date().toISOString(),
          status: 'deployed',
        },
      ])
      .select();

    if (error) {
      console.error('[save-model] Supabase error:', error);
      return NextResponse.json(
        { error: `Failed to save model: ${error.message}` },
        { status: 500 }
      );
    }

    console.log('[save-model] Model saved successfully:', data);

    return NextResponse.json({
      success: true,
      model: data?.[0],
      message: 'Model saved successfully',
    });
  } catch (error) {
    console.error('[save-model] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to save model',
      },
      { status: 500 }
    );
  }
}
