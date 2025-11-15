import { NextRequest, NextResponse } from 'next/server';
import { generateTrainingPlan, TrainingPlan } from '@/lib/groq-service';
import { E2BTrainingService } from '@/lib/e2b-training-service';
import { getSupabaseServiceRole } from '@/lib/supabase';

export const runtime = 'nodejs';
export const maxDuration = 3600; // 1 hour timeout

/**
 * Advanced Training API
 * 
 * POST /api/ai/advanced-train
 * 
 * Request:
 * {
 *   "task": "create a sentiment analysis model",
 *   "userId": "user-id",
 *   "modelId": "model-id"
 * }
 * 
 * Response:
 * {
 *   "success": true,
 *   "trainingJobId": "job-id",
 *   "plan": { ... },
 *   "message": "Training started..."
 * }
 */
export async function POST(req: NextRequest) {
  try {
    const { task, userId, modelId } = await req.json();

    if (!task || !userId || !modelId) {
      return NextResponse.json(
        { error: 'task, userId, and modelId are required' },
        { status: 400 }
      );
    }

    console.log(`🚀 Advanced training request: ${task}`);

    const supabase = getSupabaseServiceRole();

    // Create training job record
    const { data: trainingJob, error: jobError } = await (supabase
      .from('training_jobs')
      .insert as any)({
      model_id: modelId,
      user_id: userId,
      job_status: 'planning',
      total_epochs: 10,
    }).select().single();

    if (jobError) {
      console.error('Training job creation error:', jobError);
      throw new Error(`Failed to create training job: ${jobError.message}`);
    }

    console.log(`✅ Training job created: ${trainingJob.id}`);

    // Start the full training pipeline in background
    startAdvancedTraining({
      trainingJobId: trainingJob.id,
      modelId,
      userId,
      task,
    }).catch((error) => {
      console.error('❌ Advanced training pipeline error:', error);
    });

    return NextResponse.json({
      success: true,
      trainingJobId: trainingJob.id,
      message: 'Training pipeline started. Generating plan...',
    });
  } catch (error: any) {
    console.error('❌ Advanced training error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to start advanced training' },
      { status: 500 }
    );
  }
}

async function startAdvancedTraining(params: {
  trainingJobId: string;
  modelId: string;
  userId: string;
  task: string;
}) {
  const { trainingJobId, modelId, userId, task } = params;
  const supabase = getSupabaseServiceRole();

  try {
    // Step 1: Generate training plan using Groq
    console.log('📋 Step 1: Generating training plan...');
    await updateJobStatus(supabase, trainingJobId, 'planning', 'Generating training plan with AI...');

    let plan: TrainingPlan;
    try {
      plan = await generateTrainingPlan(task);
    } catch (error: any) {
      throw new Error(`Plan generation failed: ${error.message}`);
    }

    console.log('✅ Plan generated:', {
      model: plan.model.pretrained,
      dataset: plan.dataset.name,
      time: plan.estimatedTime,
    });

    // Step 2: Setup E2B sandbox and install dependencies
    console.log('📦 Step 2: Setting up E2B sandbox...');
    await updateJobStatus(supabase, trainingJobId, 'setup', 'Setting up E2B sandbox...');

    const e2bService = new E2BTrainingService();
    
    try {
      await e2bService.setupEnvironment(plan.dependencies);
    } catch (error: any) {
      throw new Error(`Environment setup failed: ${error.message}`);
    }

    // Step 3: Run training
    console.log('🏋️ Step 3: Running training...');
    await updateJobStatus(supabase, trainingJobId, 'running', 'Training model...');

    const trainingResult = await e2bService.runTraining(
      plan.trainingCode,
      async (message: string) => {
        // Update job with progress
        console.log('📊 Training progress:', message);
      }
    );

    if (!trainingResult.success) {
      throw new Error(trainingResult.error || 'Training failed');
    }

    console.log('✅ Training completed:', trainingResult.metrics);

    // Step 4: Generate deployment code
    console.log('🚀 Step 4: Preparing deployment...');
    await updateJobStatus(supabase, trainingJobId, 'deploying', 'Preparing deployment...');

    // For now, generate a mock deployment URL
    const deploymentUrl = `https://sandbox-${trainingJobId.substring(0, 8)}.e2b.dev`;

    // Step 5: Update job with completion
    console.log('📝 Step 5: Finalizing...');
    await updateJobStatus(supabase, trainingJobId, 'completed', 'Training completed!');

    await (supabase.from('training_jobs').update as any)({
      job_status: 'completed',
      completed_at: new Date().toISOString(),
      deployment_url: deploymentUrl,
      loss_value: trainingResult.metrics.loss || 0,
      accuracy: trainingResult.metrics.accuracy || 0,
      current_epoch: plan.trainingCode.includes('10') ? 10 : 5,
    }).eq('id', trainingJobId);

    await (supabase.from('ai_models').update as any)({
      training_status: 'completed',
      deployed_url: deploymentUrl,
      model_file_path: trainingResult.modelPath,
    }).eq('id', modelId);

    console.log('🎉 Advanced training pipeline completed!');
    console.log(`   Deployment URL: ${deploymentUrl}`);
    console.log(`   Metrics:`, trainingResult.metrics);

    // Cleanup
    await e2bService.cleanup();
  } catch (error: any) {
    console.error('❌ Pipeline error:', error.message);

    await (supabase.from('training_jobs').update as any)({
      job_status: 'failed',
      error_message: error.message,
      completed_at: new Date().toISOString(),
    }).eq('id', trainingJobId);

    await (supabase.from('ai_models').update as any)({
      training_status: 'failed',
    }).eq('id', modelId);
  }
}

async function updateJobStatus(
  supabase: any,
  jobId: string,
  status: string,
  message: string
) {
  console.log(`📝 Updating job status: ${status} - ${message}`);
  
  await (supabase.from('training_jobs').update as any)({
    job_status: status,
  }).eq('id', jobId);
}
