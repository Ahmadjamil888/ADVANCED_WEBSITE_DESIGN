import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseOrThrow } from '@/lib/supabase';

type TrainingJobRecord = {
  current_epoch: number | null;
  total_epochs: number | null;
  loss_value: number | null;
  accuracy: number | null;
  validation_loss: number | null;
  validation_accuracy: number | null;
  job_status: string | null;
};

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

    const supabase = getSupabaseOrThrow();
    const { data, error } = await supabase
      .from('training_jobs')
      .select('*')
      .eq('id', jobId)
      .single();

    const job = data as TrainingJobRecord | null;

    if (error || !job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    return NextResponse.json({
      currentEpoch: job.current_epoch || 0,
      totalEpochs: job.total_epochs || 10,
      loss: job.loss_value || 0,
      accuracy: job.accuracy || 0,
      validationLoss: job.validation_loss || 0,
      validationAccuracy: job.validation_accuracy || 0,
      status: job.job_status || 'unknown',
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

