import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const { prompt, model = 'mixtral-8x7b-32768' } = await request.json();

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    const baseUrl = request.nextUrl.origin;

    console.log('[orchestrate-training] Starting orchestration...');
    console.log('[orchestrate-training] Prompt:', prompt);
    console.log('[orchestrate-training] Model:', model);

    // Step 1: Generate code with Groq
    console.log('[orchestrate-training] Step 1: Generating code with Groq...');
    const groqResponse = await fetch(`${baseUrl}/api/ai/groq-generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model }),
    });

    if (!groqResponse.ok) {
      throw new Error(`Groq generation failed: ${await groqResponse.text()}`);
    }

    const groqData = await groqResponse.json();
    console.log('[orchestrate-training] Groq generation successful');
    console.log('[orchestrate-training] Code length:', groqData.code.length);
    console.log('[orchestrate-training] Requirements:', groqData.requirements);

    // Step 2: Create PyTorch sandbox
    console.log('[orchestrate-training] Step 2: Creating PyTorch sandbox...');
    const sandboxResponse = await fetch(`${baseUrl}/api/sandbox/create-pytorch-sandbox`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    if (!sandboxResponse.ok) {
      throw new Error(`Sandbox creation failed: ${await sandboxResponse.text()}`);
    }

    const sandboxData = await sandboxResponse.json();
    const sandboxId = sandboxData.sandboxId;
    console.log('[orchestrate-training] Sandbox created:', sandboxId);

    // Step 3: Train model
    console.log('[orchestrate-training] Step 3: Training model...');
    const trainingResponse = await fetch(`${baseUrl}/api/training/train-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: groqData.code,
        requirements: groqData.requirements,
        sandboxId,
      }),
    });

    if (!trainingResponse.ok) {
      throw new Error(`Model training failed: ${await trainingResponse.text()}`);
    }

    const trainingData = await trainingResponse.json();
    console.log('[orchestrate-training] Model training successful');
    console.log('[orchestrate-training] Model files:', trainingData.modelFiles);

    if (!trainingData.success) {
      throw new Error(`Training failed: ${trainingData.error}`);
    }

    // Find the trained model file
    const modelFile = trainingData.modelFiles?.[0];
    if (!modelFile) {
      throw new Error('No trained model file found');
    }

    console.log('[orchestrate-training] Using model file:', modelFile.path);

    // Step 4: Deploy to E2B
    console.log('[orchestrate-training] Step 4: Deploying to E2B...');
    const deploymentResponse = await fetch(`${baseUrl}/api/deployment/deploy-e2b`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sandboxId,
        modelPath: modelFile.path,
        modelType: groqData.modelType,
      }),
    });

    if (!deploymentResponse.ok) {
      throw new Error(`Deployment failed: ${await deploymentResponse.text()}`);
    }

    const deploymentData = await deploymentResponse.json();
    console.log('[orchestrate-training] Deployment successful');
    console.log('[orchestrate-training] Deployment URL:', deploymentData.deploymentUrl);

    return NextResponse.json({
      success: true,
      steps: {
        codeGeneration: {
          status: 'completed',
          model: groqData.model,
          codeLength: groqData.code.length,
          modelType: groqData.modelType,
          requirements: groqData.requirements,
        },
        sandboxCreation: {
          status: 'completed',
          sandboxId,
        },
        modelTraining: {
          status: 'completed',
          output: trainingData.output,
          modelFiles: trainingData.modelFiles,
        },
        deployment: {
          status: 'completed',
          deploymentUrl: deploymentData.deploymentUrl,
          endpoints: deploymentData.endpoints,
        },
      },
      deploymentUrl: deploymentData.deploymentUrl,
      sandboxId,
      modelType: groqData.modelType,
      message: 'AI model successfully generated, trained, and deployed!',
    });
  } catch (error) {
    console.error('[orchestrate-training] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Orchestration failed',
      },
      { status: 500 }
    );
  }
}
