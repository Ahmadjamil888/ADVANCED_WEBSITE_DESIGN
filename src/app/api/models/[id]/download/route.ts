import { NextResponse, type NextRequest } from 'next/server';
import { getSupabaseOrThrow, type Database } from '@/lib/supabase';

type AIModelRow = Database['public']['Tables']['ai_models']['Row'];

export async function GET(
  _req: NextRequest,
  context: { params: Promise<Record<string, string | string[] | undefined>> }
) {
  try {
    const params = await context.params;
    const idParam = params?.id;
    const modelId = Array.isArray(idParam) ? idParam[0] : idParam;

    if (!modelId) {
      return NextResponse.json({ error: 'Model ID is required' }, { status: 400 });
    }

    const supabase = getSupabaseOrThrow();
    const { data, error } = await supabase
      .from('ai_models')
      .select('model_file_path, model_file_format, name')
      .eq('id', modelId)
      .single();

    const model = data as Pick<AIModelRow, 'model_file_path' | 'model_file_format' | 'name'> | null;

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

