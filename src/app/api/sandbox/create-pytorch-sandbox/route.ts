import { NextRequest, NextResponse } from 'next/server';

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
        await (global.activePyTorchSandbox as any).kill();
      } catch (e) {
        console.error('[create-pytorch-sandbox] Error stopping existing sandbox:', e);
      }
    }

    // Dynamic import for E2B SDK - handle CommonJS
    let Sandbox: any;
    try {
      const e2bModule = await import('@e2b/code-interpreter');
      // Try different export patterns
      Sandbox = e2bModule.Sandbox || e2bModule.default?.Sandbox || e2bModule.default;
      
      if (!Sandbox || typeof Sandbox.create !== 'function') {
        throw new Error('Sandbox.create is not available');
      }
    } catch (importError) {
      console.error('[create-pytorch-sandbox] Import error:', importError);
      throw new Error(`Failed to import E2B SDK: ${importError instanceof Error ? importError.message : 'Unknown error'}`);
    }

    // Create new E2B sandbox
    console.log('[create-pytorch-sandbox] Creating sandbox with Sandbox.create...');
    const sandbox: any = await Sandbox.create({
      apiKey: process.env.E2B_API_KEY,
      timeoutMs: 30 * 60 * 1000, // 30 minutes timeout
    });

    if (!sandbox) {
      throw new Error('Sandbox creation returned null or undefined');
    }

    global.activePyTorchSandbox = sandbox;
    global.sandboxId = sandbox.sandboxId;

    console.log('[create-pytorch-sandbox] Sandbox created:', sandbox.sandboxId);
    console.log('[create-pytorch-sandbox] Sandbox type:', typeof sandbox);
    console.log('[create-pytorch-sandbox] Sandbox methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(sandbox)));

    // Install PyTorch and dependencies
    console.log('[create-pytorch-sandbox] Installing PyTorch and dependencies...');

    if (typeof sandbox.runPython !== 'function') {
      throw new Error(`runPython is not a function. Available methods: ${Object.getOwnPropertyNames(Object.getPrototypeOf(sandbox)).join(', ')}`);
    }

    const installResult = await sandbox.runPython(`
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
