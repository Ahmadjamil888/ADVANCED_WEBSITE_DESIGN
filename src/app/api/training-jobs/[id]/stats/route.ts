import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServiceRole } from '@/lib/supabase';

type TrainingJobRecord = {
  id: string;
  model_id: string;
  current_epoch: number | null;
  total_epochs: number | null;
  loss_value: number | null;
  accuracy: number | null;
  validation_loss: number | null;
  validation_accuracy: number | null;
  job_status: string | null;
  deployment_url?: string | null;
  error_message?: string | null;
};

export const runtime = 'nodejs';

export async function GET(
  _req: NextRequest,
  context: { params: Promise<Record<string, string | string[] | undefined>> }
) {
  try {
    const params = await context.params;
    const idParam = params?.id;
    const jobId = Array.isArray(idParam) ? idParam[0] : idParam;

    if (!jobId) {
      return NextResponse.json({ error: 'Job ID is required' }, { status: 400 });
    }

    const supabase = getSupabaseServiceRole();
    const { data, error } = await (supabase
      .from('training_jobs')
      .select('*')
      .eq('id', jobId)
      .single() as any);

    const job = data as TrainingJobRecord | null;

    if (error || !job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    // Get associated model to check for deployment URL
    const { data: model, error: modelError } = await (supabase
      .from('ai_models')
      .select('*')
      .eq('id', job.model_id)
      .single() as any);

    const deploymentUrl = model?.deployed_url || model?.deployment_url || job.deployment_url;

    return NextResponse.json({
      id: job.id,
      currentEpoch: job.current_epoch || 0,
      totalEpochs: job.total_epochs || 10,
      loss: job.loss_value || 0,
      accuracy: job.accuracy || 0,
      validationLoss: job.validation_loss || 0,
      validationAccuracy: job.validation_accuracy || 0,
      status: job.job_status || 'unknown',
      deployment_url: deploymentUrl,
      error_message: job.error_message,
    });
  } catch (error: any) {
    console.error('Error fetching training job stats:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

