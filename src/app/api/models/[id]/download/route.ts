import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseOrThrow } from '@/lib/supabase';

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = getSupabaseOrThrow();
    const { data: model, error } = await supabase
      .from('ai_models')
      .select('model_file_path, model_file_format, name')
      .eq('id', params.id)
      .single();

    if (error || !model?.model_file_path) {
      return NextResponse.json({ error: 'Model not found' }, { status: 404 });
    }

    // In production, fetch from storage (S3, etc.)
    // For now, return placeholder
    return NextResponse.json({
      downloadUrl: `/api/storage/${model.model_file_path}`,
      filename: `${model.name || 'model'}.${model.model_file_format || 'pth'}`,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

