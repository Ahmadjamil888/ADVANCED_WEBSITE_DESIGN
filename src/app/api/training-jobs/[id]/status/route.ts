import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServiceRole } from '@/lib/supabase';

export const runtime = 'nodejs';

/**
 * GET /api/training-jobs/[id]/status
 * Returns the status of a training job including deployment URL
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const trainingJobId = id;

    if (!trainingJobId) {
      return NextResponse.json(
        { error: 'Training job ID is required' },
        { status: 400 }
      );
    }

    const supabase = getSupabaseServiceRole();

    // Get training job details
    const { data: trainingJob, error: jobError } = await (supabase
      .from('training_jobs')
      .select('*')
      .eq('id', trainingJobId)
      .single() as any);

    if (jobError || !trainingJob) {
      return NextResponse.json(
        { error: 'Training job not found' },
        { status: 404 }
      );
    }

    // Get associated model to check for deployment URL
    const { data: model, error: modelError } = await (supabase
      .from('ai_models')
      .select('*')
      .eq('id', trainingJob.model_id)
      .single() as any);

    if (modelError) {
      console.error('Error fetching model:', modelError);
    }

    return NextResponse.json({
      id: trainingJob.id,
      job_status: trainingJob.job_status,
      created_at: trainingJob.created_at,
      started_at: trainingJob.started_at,
      completed_at: trainingJob.completed_at,
      current_epoch: trainingJob.current_epoch,
      total_epochs: trainingJob.total_epochs,
      progress_percentage: trainingJob.progress_percentage,
      loss_value: trainingJob.loss_value,
      accuracy: trainingJob.accuracy,
      error_message: trainingJob.error_message,
      // Include deployment URL from model if available
      deployment_url: model?.deployed_url || model?.deployment_url,
      model_file_path: model?.model_file_path,
    });
  } catch (error: any) {
    console.error('Error fetching training job status:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch training job status' },
      { status: 500 }
    );
  }
}
