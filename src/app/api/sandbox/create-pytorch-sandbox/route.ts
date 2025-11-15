import { NextRequest, NextResponse } from 'next/server';
// @ts-ignore - e2b types not available
import { Sandbox } from '@e2b/code-interpreter';

declare global {
  var activePyTorchSandbox: any;
  var sandboxId: string;
}

export async function POST(request: NextRequest) {
  try {
    console.log('[create-pytorch-sandbox] Creating E2B sandbox for PyTorch training...');

    if (!process.env.E2B_API_KEY) {
      return NextResponse.json(
        { error: 'E2B_API_KEY is not configured' },
        { status: 500 }
      );
    }

    // Stop existing sandbox if any
    if (global.activePyTorchSandbox) {
      try {
        console.log('[create-pytorch-sandbox] Stopping existing sandbox...');
        await global.activePyTorchSandbox.kill();
      } catch (e) {
        console.error('[create-pytorch-sandbox] Error stopping existing sandbox:', e);
      }
    }

    // Create new E2B sandbox
    const sandbox: any = await (Sandbox as any).create({
      apiKey: process.env.E2B_API_KEY,
      timeoutMs: 30 * 60 * 1000, // 30 minutes timeout
    });

    global.activePyTorchSandbox = sandbox;
    global.sandboxId = sandbox.sandboxId;

    console.log('[create-pytorch-sandbox] Sandbox created:', sandbox.sandboxId);

    // Install PyTorch and dependencies
    console.log('[create-pytorch-sandbox] Installing PyTorch and dependencies...');

    const installResult = await (sandbox as any).runPython(`
import subprocess
import sys

packages = [
    'torch',
    'torchvision',
    'torchaudio',
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'requests',
    'tqdm',
    'jupyter',
    'uvicorn'
]

for package in packages:
    print(f'Installing {package}...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print('All packages installed successfully!')
`);

    console.log('[create-pytorch-sandbox] Installation output:', installResult.stdout);

    if (installResult.error) {
      console.error('[create-pytorch-sandbox] Installation error:', installResult.error);
    }

    return NextResponse.json({
      success: true,
      sandboxId: sandbox.sandboxId,
      message: 'PyTorch sandbox created successfully',
    });
  } catch (error) {
    console.error('[create-pytorch-sandbox] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to create sandbox',
      },
      { status: 500 }
    );
  }
}
