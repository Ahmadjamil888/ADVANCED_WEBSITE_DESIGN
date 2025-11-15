import { NextRequest, NextResponse } from 'next/server';
// @ts-ignore - e2b types not available
import { Sandbox } from '@e2b/code-interpreter';

declare global {
  var activePyTorchSandbox: any;
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

    let sandbox: any = global.activePyTorchSandbox;

    // If no active sandbox or different sandbox ID, create/connect to sandbox
    if (!sandbox || global.sandboxId !== sandboxId) {
      console.log('[train-model] Creating new sandbox for training...');
      sandbox = await (Sandbox as any).create({
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

      const installResult = await sandbox.runPython(installCode);
      console.log('[train-model] Installation output:', installResult.stdout);
      if (installResult.error) {
        console.error('[train-model] Installation error:', installResult.error);
      }
    }

    // Run the training code
    console.log('[train-model] Running training code...');
    const result = await sandbox.runPython(code);

    console.log('[train-model] Training output:', result.stdout);

    if (result.error) {
      console.error('[train-model] Training error:', result.error);
      return NextResponse.json(
        {
          success: false,
          error: result.error,
          stdout: result.stdout,
        },
        { status: 500 }
      );
    }

    // List files in sandbox to find trained model
    console.log('[train-model] Listing sandbox files...');
    const filesResult = await sandbox.runPython(`
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
      const output = filesResult.stdout.trim();
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
      output: result.stdout,
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
