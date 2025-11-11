import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseOrThrow } from '@/lib/supabase';

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = getSupabaseOrThrow();
    const { data: job, error } = await supabase
      .from('training_jobs')
      .select('*')
      .eq('id', params.id)
      .single();

    if (error) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    return NextResponse.json({
      currentEpoch: job.current_epoch || 0,
      totalEpochs: job.total_epochs || 10,
      loss: job.loss_value || 0,
      accuracy: job.accuracy || 0,
      validationLoss: job.validation_loss || 0,
      validationAccuracy: job.validation_accuracy || 0,
      status: job.job_status,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

