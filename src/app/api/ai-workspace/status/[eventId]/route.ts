import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ eventId: string }> }
) {
  try {
    const { eventId } = await params

    if (!eventId) {
      return NextResponse.json({ error: 'Event ID is required' }, { status: 400 })
    }

    if (!supabase) {
      return NextResponse.json({ error: 'Database not available' }, { status: 500 })
    }

    // Check if model is ready by looking for it in the database
    const { data: models, error } = await supabase
      .from('ai_models')
      .select(`
        *,
        training_jobs (
          job_status,
          progress_percentage,
          sandbox_session_id
        )
      `)
      .eq('metadata->>eventId', eventId)
      .order('created_at', { ascending: false })
      .limit(1)

    if (error) {
      console.error('Database error:', error)
      return NextResponse.json({ error: 'Database query failed' }, { status: 500 })
    }

    if (!models || models.length === 0) {
      return NextResponse.json({ 
        status: 'processing',
        message: 'Model generation in progress...',
        ready: false
      })
    }

    const model = models[0]
    const trainingJob = model.training_jobs?.[0]

    return NextResponse.json({
      status: model.training_status,
      ready: model.training_status === 'ready',
      model: {
        id: model.id,
        name: model.name,
        type: model.model_type,
        framework: model.framework,
        dataset: model.dataset_name,
        created_at: model.created_at,
        file_structure: model.file_structure,
        huggingface_repo: model.huggingface_repo
      },
      training_job: trainingJob ? {
        status: trainingJob.job_status,
        progress: trainingJob.progress_percentage,
        sandbox_id: trainingJob.sandbox_session_id
      } : null,
      eventId
    })

  } catch (error: any) {
    console.error('Status check error:', error)
    
    return NextResponse.json(
      { error: `Status check failed: ${error.message}` },
      { status: 500 }
    )
  }
}