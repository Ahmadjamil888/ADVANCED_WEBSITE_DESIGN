import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const supabase = createClient(supabaseUrl, supabaseKey);

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params;

    if (!eventId) {
      return NextResponse.json(
        { error: 'Missing event ID' },
        { status: 400 }
      );
    }

    // Get the training job from the database
    const { data: job, error } = await supabase
      .from('training_jobs')
      .select('*')
      .eq('id', eventId)
      .single();

    if (error || !job) {
      return NextResponse.json(
        { error: 'Training job not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      status: job.status,
      progress: job.progress || 0,
      message: job.message || '',
      e2bUrl: job.e2b_url || null,
      modelFiles: job.model_files || [],
      startedAt: job.started_at,
      completedAt: job.completed_at,
      error: job.error_message || null
    });

  } catch (error) {
    console.error('Error fetching training status:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params;
    const updates = await request.json();

    if (!eventId) {
      return NextResponse.json(
        { error: 'Missing event ID' },
        { status: 400 }
      );
    }

    // Update the training job in the database
    const { data: job, error } = await supabase
      .from('training_jobs')
      .update(updates)
      .eq('id', eventId)
      .select()
      .single();

    if (error) {
      console.error('Error updating training job:', error);
      return NextResponse.json(
        { error: 'Failed to update training job' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      job
    });

  } catch (error) {
    console.error('Error updating training status:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export const runtime = 'nodejs';
