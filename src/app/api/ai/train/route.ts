import { NextRequest, NextResponse } from 'next/server';
import { E2BManager } from '@/lib/e2b';
import { AIClient } from '@/lib/ai/client';
import { AI_MODELS } from '@/lib/ai/models';
import { getSupabaseOrThrow } from '@/lib/supabase';

export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  try {
    const { modelId, userId, prompt, trainingMode, datasetPath, modelPath, extraInstructions } = await req.json();

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 });
    }

    if (!modelId) {
      return NextResponse.json({ error: 'Model ID is required' }, { status: 400 });
    }

    const supabase = getSupabaseOrThrow();

    // Create training job
    const { data: trainingJob, error: jobError } = await (supabase
      .from('training_jobs')
      .insert as any)({
      model_id: modelId,
      user_id: userId,
      job_status: 'queued',
      total_epochs: 10,
    }).select().single();

    if (jobError) {
      console.error('Training job creation error:', jobError);
      throw new Error(`Failed to create training job: ${jobError.message}`);
    }

    // Update model status
    await (supabase.from('ai_models').update as any)({
      training_status: 'queued',
    }).eq('id', modelId);

    // Start training in background (don't await)
    trainModelInBackground({
      modelId,
      trainingJobId: trainingJob.id,
      prompt,
      trainingMode,
      datasetPath,
      modelPath,
      extraInstructions,
    }).catch(console.error);

    return NextResponse.json({ success: true, trainingJobId: trainingJob.id });
  } catch (error: any) {
    console.error('Training start error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

async function trainModelInBackground(params: {
  modelId: string;
  trainingJobId: string;
  prompt: string;
  trainingMode: string;
  datasetPath?: string;
  modelPath?: string;
  extraInstructions?: string;
}) {
  const { modelId, trainingJobId, prompt, trainingMode, datasetPath, modelPath, extraInstructions } = params;
  const supabase = getSupabaseOrThrow();

  try {
    // Scrape dataset/model from HF/Kaggle if not provided
    let finalDatasetPath = datasetPath;
    let finalModelPath = modelPath;

    if (!finalDatasetPath || (trainingMode === 'fine_tune' && !finalModelPath)) {
      const scraped = await scrapeResources(prompt, trainingMode);
      if (!finalDatasetPath && scraped.dataset) finalDatasetPath = scraped.dataset;
      if (!finalModelPath && scraped.model) finalModelPath = scraped.model;
    }

    // Update job status
    await (supabase.from('training_jobs').update as any)({
      job_status: 'running',
      started_at: new Date().toISOString(),
    }).eq('id', trainingJobId);

    // Generate code using AI
    const aiClient = new AIClient('groq', 'llama-3.3-70b-versatile');
    let fullCode = '';
    for await (const chunk of aiClient.streamCompletion([
      {
        role: 'system',
        content: `You are an AI code generator. Generate complete, runnable Python code for training ${trainingMode === 'from_scratch' ? 'a new' : 'fine-tuning an existing'} AI model based on: ${prompt}. Include train.py with real-time epoch logging, model saving as .pth, and accuracy calculation.`,
      },
      { role: 'user', content: prompt + (extraInstructions ? `\n\nAdditional requirements: ${extraInstructions}` : '') },
    ])) {
      if (!chunk.done) {
        fullCode += chunk.content;
      }
    }

    // Parse and write files to E2B
    const e2b = new E2BManager();
    const sandboxId = await e2b.createSandbox();

    const files = parseFilesFromCode(fullCode);
    await e2b.writeFiles(files);

    // Install dependencies
    if (files['requirements.txt']) {
      await e2b.installDependencies();
    }

    // Run training with real-time stats
    if (files['train.py']) {
      await e2b.runCommand(
        'python /home/user/train.py',
        async (stdout: string) => {
          // Parse epoch stats from stdout
          const epochMatch = stdout.match(/Epoch (\d+)\/(\d+).*Loss: ([\d.]+).*Accuracy: ([\d.]+)/);
          if (epochMatch) {
            const [, epoch, total, loss, accuracy] = epochMatch;
            await (supabase.from('training_epochs').insert as any)({
              training_job_id: trainingJobId,
              epoch_number: parseInt(epoch),
              loss: parseFloat(loss),
              accuracy: parseFloat(accuracy),
            });
            await (supabase.from('training_jobs').update as any)({
              current_epoch: parseInt(epoch),
              loss_value: parseFloat(loss),
              accuracy: parseFloat(accuracy),
              progress_percentage: Math.round((parseInt(epoch) / parseInt(total)) * 100),
            }).eq('id', trainingJobId);
          }
        },
        async (stderr: string) => {
          console.error('Training error:', stderr);
        }
      );
    }

    // Update completion
    await (supabase.from('training_jobs').update as any)({
      job_status: 'completed',
      completed_at: new Date().toISOString(),
    }).eq('id', trainingJobId);

    await (supabase.from('ai_models').update as any)({
      training_status: 'completed',
      model_file_path: `/home/user/model.pth`,
      model_file_format: 'pth',
    }).eq('id', modelId);

    await e2b.close();
  } catch (error: any) {
    console.error('Training error:', error);
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

async function scrapeResources(prompt: string, trainingMode: string): Promise<{ dataset?: string; model?: string }> {
  const result: { dataset?: string; model?: string } = {};

  // Scrape dataset if needed
  if (trainingMode === 'from_scratch') {
    try {
      const hfRes = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/scrape/huggingface`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, type: 'dataset' }),
      });
      const hfData = await hfRes.json();
      if (hfData.success) {
        result.dataset = hfData.repo;
      } else {
        // Try Kaggle
        const kgRes = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/scrape/kaggle`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
        });
        const kgData = await kgRes.json();
        if (kgData.success && !kgData.requiresManualSelection) {
          result.dataset = kgData.repo;
        }
      }
    } catch (error) {
      console.error('Dataset scraping error:', error);
    }
  }

  // Scrape model for fine-tuning
  if (trainingMode === 'fine_tune') {
    try {
      const hfRes = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/scrape/huggingface`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, type: 'model' }),
      });
      const hfData = await hfRes.json();
      if (hfData.success) {
        result.model = hfData.repo;
      }
    } catch (error) {
      console.error('Model scraping error:', error);
    }
  }

  return result;
}

function parseFilesFromCode(code: string): Record<string, string> {
  const files: Record<string, string> = {};
  const fileRegex = /<file path="([^"]+)">([\s\S]*?)<\/file>/g;
  let match;
  while ((match = fileRegex.exec(code)) !== null) {
    files[match[1]] = match[2].trim();
  }
  return files;
}

