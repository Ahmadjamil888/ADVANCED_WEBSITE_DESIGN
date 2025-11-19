import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { getAuth } from '@clerk/nextjs/server';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
);

export async function GET(request: NextRequest) {
  try {
    const { userId } = getAuth(request);

    if (!userId) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Fetch models from Supabase
    const { data: models, error } = await supabase
      .from('trained_models')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(`Failed to fetch models: ${error.message}`);
    }

    // Transform data for frontend
    const transformedModels = (models || []).map((model: any) => ({
      id: model.id,
      name: model.name,
      type: model.model_type,
      description: model.description,
      datasetSource: model.dataset_source,
      createdAt: model.created_at,
      finalLoss: model.final_loss,
      finalAccuracy: model.final_accuracy,
      epochs: model.epochs_trained,
      stats: model.stats || [],
      sandboxUrl: model.sandbox_url,
      modelPath: model.model_path,
    }));

    return NextResponse.json({ models: transformedModels });
  } catch (error) {
    console.error('Failed to list models:', error);
    return NextResponse.json(
      { error: 'Failed to list models' },
      { status: 500 }
    );
  }
}
