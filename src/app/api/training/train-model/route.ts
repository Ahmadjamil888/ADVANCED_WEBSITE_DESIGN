import { NextRequest, NextResponse } from 'next/server';

declare global {
  var activePyTorchSandbox: any;
  var sandboxId: string;
}

export async function POST(request: NextRequest) {
  try {
    const { code, requirements = [], sandboxId } = await request.json();

    if (!code) {
      return NextResponse.json(
        { error: 'Code is required' },
        { status: 400 }
      );
    }

    if (!process.env.E2B_API_KEY) {
      return NextResponse.json(
        { error: 'E2B_API_KEY is not configured' },
        { status: 500 }
      );
    }

    console.log('[train-model] Starting model training...');
    console.log('[train-model] Sandbox ID:', sandboxId);
    console.log('[train-model] Requirements:', requirements);

    // Import E2B SDK
    let Sandbox: any;
    try {
      const e2bModule: any = await import('@e2b/code-interpreter');
      Sandbox = e2bModule.Sandbox || e2bModule.default;
      
      if (!Sandbox) {
        throw new Error('Sandbox class not found');
      }
    } catch (importError) {
      console.error('[train-model] Import error:', importError);
      throw new Error(`Failed to import E2B SDK: ${importError instanceof Error ? importError.message : 'Unknown error'}`);
    }

    let sandbox: any = global.activePyTorchSandbox;

    // If no active sandbox or different sandbox ID, create/connect to sandbox
    if (!sandbox || global.sandboxId !== sandboxId) {
      console.log('[train-model] Creating new sandbox for training...');
      sandbox = await Sandbox.create({
        apiKey: process.env.E2B_API_KEY,
        timeoutMs: 60 * 60 * 1000, // 1 hour timeout for training
      });
      global.activePyTorchSandbox = sandbox;
      global.sandboxId = sandbox.sandboxId;
    }

    // Install additional requirements
    if (requirements.length > 0) {
      console.log('[train-model] Installing requirements...');
      const installCode = `
import subprocess
import sys

packages = ${JSON.stringify(requirements)}

for package in packages:
    try:
        print(f'Installing {package}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        print(f'[OK] {package} installed')
    except Exception as e:
        print(f'[ERROR] Error installing {package}: {e}')
`;

      const installResult = await sandbox.runCode(installCode);
      console.log('[train-model] Installation output:', installResult.logs?.stdout || installResult);
      if (installResult.error) {
        console.error('[train-model] Installation error:', installResult.error);
      }
    }

    // Run the training code
    console.log('[train-model] Running training code...');
    const result = await sandbox.runCode(code);

    console.log('[train-model] Training output:', result.logs?.stdout || result);

    if (result.error) {
      console.error('[train-model] Training error:', result.error);
      return NextResponse.json(
        {
          success: false,
          error: result.error,
          stdout: result.logs?.stdout,
        },
        { status: 500 }
      );
    }

    // List files in sandbox to find trained model
    console.log('[train-model] Listing sandbox files...');
    const filesResult = await sandbox.runCode(`
import os
import json

files = []
for root, dirs, filenames in os.walk('/tmp'):
    for filename in filenames:
        if filename.endswith(('.pt', '.pth', '.pkl', '.joblib')):
            filepath = os.path.join(root, filename)
            size = os.path.getsize(filepath)
            files.append({'path': filepath, 'size': size})

print(json.dumps(files))
`);

    let modelFiles = [];
    try {
      const output = (filesResult.logs?.stdout || filesResult).trim();
      const jsonMatch = output.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        modelFiles = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      console.error('[train-model] Error parsing model files:', e);
    }

    console.log('[train-model] Found model files:', modelFiles);

    return NextResponse.json({
      success: true,
      sandboxId: sandbox.sandboxId,
      output: result.logs?.stdout,
      modelFiles,
      message: 'Model training completed successfully',
    });
  } catch (error) {
    console.error('[train-model] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to train model',
      },
      { status: 500 }
    );
  }
}
