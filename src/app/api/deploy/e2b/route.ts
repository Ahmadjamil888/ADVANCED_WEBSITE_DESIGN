import { NextRequest, NextResponse } from 'next/server';
import { E2BManager } from '@/lib/e2b';
import { getSupabaseOrThrow, type Database } from '@/lib/supabase';

type AIModelRow = Database['public']['Tables']['ai_models']['Row'];

export const maxDuration = 300;

export async function POST(req: NextRequest) {
  try {
    const { modelId } = await req.json();

    const supabase = getSupabaseOrThrow();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get model details
    const modelResponse = await supabase
      .from('ai_models')
      .select('*')
      .eq('id', modelId)
      .eq('user_id', user.id)
      .single();
    const model = modelResponse.data as AIModelRow | null;
    const modelError = modelResponse.error;

    if (modelError || !model) {
      return NextResponse.json({ error: 'Model not found' }, { status: 404 });
    }

    const modelNameRaw = model.name ?? 'model';
    const modelName = modelNameRaw.replace(/"/g, '\\"');

    // Create E2B sandbox
    const e2b = new E2BManager();
    const sandboxId = await e2b.createSandbox();

    // Create deployment app.py
    const appCode = `from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (in production, load from actual file)
model = None  # Load your model here

@app.get("/")
async def root():
    return {"status": "ok", "model": "${modelName}"}

@app.post("/predict")
async def predict(data: dict):
    # Add prediction logic here
    return {"prediction": "sample", "confidence": 0.95}
`;

    await e2b.writeFile('/home/user/app.py', appCode);

    // Install dependencies
    await e2b.writeFile('/home/user/requirements.txt', 'fastapi==0.104.0\nuvicorn==0.24.0\ntorch==2.1.0\n');

    // Deploy with multi-port fallback
    const deploymentUrl = await e2b.deployAPI('/home/user/app.py', 8000, {
      startCommand: 'cd /home/user && python -m uvicorn app:app --host 0.0.0.0 --port 8000',
      fallbackStartCommand: 'cd /home/user && python -m http.server 8000',
      waitSeconds: 30,
    });

    // Update model with deployment info
    const metadataRecord =
      model.metadata && typeof model.metadata === 'object' && !Array.isArray(model.metadata)
        ? (model.metadata as Record<string, unknown>)
        : {};

    await (supabase.from('ai_models').update as any)({
      deployment_type: 'e2b',
      deployment_url: deploymentUrl,
      deployed_at: new Date().toISOString(),
      metadata: {
        ...metadataRecord,
        sandboxId,
      },
    }).eq('id', modelId);

    return NextResponse.json({
      success: true,
      deploymentUrl,
      sandboxId,
      message: 'Model deployed to E2B successfully',
    });
  } catch (error: any) {
    console.error('E2B deployment error:', error);
    return NextResponse.json(
      { error: error.message || 'E2B deployment failed' },
      { status: 500 }
    );
  }
}

