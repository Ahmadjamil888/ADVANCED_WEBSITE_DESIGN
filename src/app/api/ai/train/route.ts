import { NextRequest, NextResponse } from 'next/server';
import { E2BManager } from '@/lib/e2b';
import { AIClient } from '@/lib/ai/client';
import { AI_MODELS } from '@/lib/ai/models';
import { getSupabaseServiceRole } from '@/lib/supabase';

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

    let supabase;
    try {
      supabase = getSupabaseServiceRole();
    } catch (error: any) {
      return NextResponse.json({ 
        error: 'Service role not configured. Add SUPABASE_SERVICE_ROLE_KEY to environment variables.' 
      }, { status: 500 });
    }

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
  const supabase = getSupabaseServiceRole();

  try {
    console.log(`🚀 Starting training for model ${modelId}, job ${trainingJobId}`);
    
    // Scrape dataset/model from HF/Kaggle if not provided
    let finalDatasetPath = datasetPath;
    let finalModelPath = modelPath;

    if (!finalDatasetPath || (trainingMode === 'fine_tune' && !finalModelPath)) {
      console.log('📦 Scraping resources...');
      const scraped = await scrapeResources(prompt, trainingMode);
      if (!finalDatasetPath && scraped.dataset) finalDatasetPath = scraped.dataset;
      if (!finalModelPath && scraped.model) finalModelPath = scraped.model;
      console.log('✅ Resources scraped:', { finalDatasetPath, finalModelPath });
    }

    // Update job status
    console.log('📝 Updating job status to running...');
    await (supabase.from('training_jobs').update as any)({
      job_status: 'running',
      started_at: new Date().toISOString(),
    }).eq('id', trainingJobId);

    // Generate code using AI
    console.log('🤖 Generating code with AI...');
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
    console.log('✅ Code generated, length:', fullCode.length);

    // Parse and write files to E2B
    console.log('📝 Creating E2B sandbox...');
    const e2b = new E2BManager();
    const sandboxId = await e2b.createSandbox();
    console.log('✅ Sandbox created:', sandboxId);

    console.log('📄 Parsing files from generated code...');
    const files = parseFilesFromCode(fullCode);
    console.log('✅ Files parsed:', Object.keys(files));
    
    console.log('💾 Writing files to sandbox...');
    await e2b.writeFiles(files);
    console.log('✅ Files written');

    // Install dependencies
    if (files['requirements.txt']) {
      console.log('📦 Installing dependencies...');
      await e2b.installDependencies();
      console.log('✅ Dependencies installed');
    }

    // Run training with real-time stats
    if (files['train.py']) {
      console.log('🏋️ Running training...');
      await e2b.runCommand(
        'python /home/user/train.py',
        async (stdout: string) => {
          console.log('[training]', stdout);
          // Parse epoch stats from stdout
          const epochMatch = stdout.match(/Epoch (\d+)\/(\d+).*Loss: ([\d.]+).*Accuracy: ([\d.]+)/);
          if (epochMatch) {
            const [, epoch, total, loss, accuracy] = epochMatch;
            console.log(`📊 Epoch ${epoch}/${total}: Loss=${loss}, Accuracy=${accuracy}`);
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
          console.error('[training error]', stderr);
        }
      );
      console.log('✅ Training completed');
    }

    // Deploy the app to E2B and get the deployment URL
    let deploymentUrl = '';
    try {
      console.log('🚀 Deploying app to E2B...');
      deploymentUrl = await e2b.deployAPI('/home/user/app.py', 8000, {
        startCommand: `cd /home/user && python -m uvicorn app:app --host 0.0.0.0 --port 8000`,
        fallbackStartCommand: `cd /home/user && python -m http.server 8000`,
        waitSeconds: 30,
      });
      console.log('✅ Model deployed at:', deploymentUrl);
    } catch (error: any) {
      console.error('❌ Deployment error:', error);
      // Continue even if deployment fails
    }

    // Update completion with deployment URL
    console.log('📝 Updating job status to completed...');
    await (supabase.from('training_jobs').update as any)({
      job_status: 'completed',
      completed_at: new Date().toISOString(),
      deployment_url: deploymentUrl,
    }).eq('id', trainingJobId);

    await (supabase.from('ai_models').update as any)({
      training_status: 'completed',
      model_file_path: `/home/user/model.pth`,
      model_file_format: 'pth',
      deployed_url: deploymentUrl,
    }).eq('id', modelId);

    console.log('🎉 Training job completed successfully!');
    await e2b.close();
  } catch (error: any) {
    console.error('❌ Training error:', error);
    console.error('Error stack:', error.stack);
    const errorMessage = error.message || 'Unknown error occurred';
    
    try {
      await (supabase.from('training_jobs').update as any)({
        job_status: 'failed',
        error_message: errorMessage,
        completed_at: new Date().toISOString(),
      }).eq('id', trainingJobId);
      
      await (supabase.from('ai_models').update as any)({
        training_status: 'failed',
      }).eq('id', modelId);
    } catch (dbError) {
      console.error('❌ Failed to update database with error:', dbError);
    }
  }
}

async function scrapeResources(prompt: string, trainingMode: string): Promise<{ dataset?: string; model?: string }> {
  const result: { dataset?: string; model?: string } = {};

  // Scrape dataset if needed (non-blocking - errors are gracefully handled)
  if (trainingMode === 'from_scratch') {
    try {
      const appUrl = process.env.NEXT_PUBLIC_APP_URL;
      if (!appUrl) {
        console.warn('NEXT_PUBLIC_APP_URL not set, skipping dataset scraping');
        return result;
      }

      const hfRes = await fetch(`${appUrl}/api/scrape/huggingface`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, type: 'dataset' }),
        signal: AbortSignal.timeout(10000), // 10 second timeout
      } as any);
      const hfData = await hfRes.json();
      if (hfData.success) {
        result.dataset = hfData.repo;
      } else {
        // Try Kaggle
        const kgRes = await fetch(`${appUrl}/api/scrape/kaggle`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
          signal: AbortSignal.timeout(10000),
        } as any);
        const kgData = await kgRes.json();
        if (kgData.success && !kgData.requiresManualSelection) {
          result.dataset = kgData.repo;
        }
      }
    } catch (error) {
      console.warn('Dataset scraping failed (non-blocking):', error);
      // Continue without dataset - training can proceed
    }
  }

  // Scrape model for fine-tuning (non-blocking)
  if (trainingMode === 'fine_tune') {
    try {
      const appUrl = process.env.NEXT_PUBLIC_APP_URL;
      if (!appUrl) {
        console.warn('NEXT_PUBLIC_APP_URL not set, skipping model scraping');
        return result;
      }

      const hfRes = await fetch(`${appUrl}/api/scrape/huggingface`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, type: 'model' }),
        signal: AbortSignal.timeout(10000),
      } as any);
      const hfData = await hfRes.json();
      if (hfData.success) {
        result.model = hfData.repo;
      }
    } catch (error) {
      console.warn('Model scraping failed (non-blocking):', error);
      // Continue without model - training can proceed
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

