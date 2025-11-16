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

    // Import E2B SDK
    let Sandbox: any;
    try {
      const e2bModule: any = await import('@e2b/code-interpreter');
      Sandbox = e2bModule.Sandbox || e2bModule.default;
      
      if (!Sandbox) {
        throw new Error('Sandbox class not found in E2B module');
      }
    } catch (importError) {
      console.error('[create-pytorch-sandbox] Import error:', importError);
      throw new Error(`Failed to import E2B SDK: ${importError instanceof Error ? importError.message : 'Unknown error'}`);
    }

    // Create new E2B sandbox
    console.log('[create-pytorch-sandbox] Creating sandbox...');
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

    if (typeof sandbox.runCode !== 'function') {
      throw new Error(`runCode is not a function. Available methods: ${Object.getOwnPropertyNames(Object.getPrototypeOf(sandbox)).join(', ')}`);
    }

    const installResult = await sandbox.runCode(`
import subprocess
import sys
import time

print("=" * 70)
print("🚀 INSTALLING PYTORCH AND ALL DEPENDENCIES")
print("=" * 70)

# Core ML libraries
core_packages = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'torchaudio>=2.0.0',
]

# Data processing
data_packages = [
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'scipy>=1.10.0',
    'scikit-learn>=1.3.0',
]

# HuggingFace ecosystem
hf_packages = [
    'transformers>=4.30.0',
    'datasets>=2.13.0',
    'huggingface-hub>=0.16.0',
    'tokenizers>=0.13.0',
]

# Data sources
data_source_packages = [
    'kaggle>=1.5.0',
    'requests>=2.31.0',
]

# Visualization & monitoring
viz_packages = [
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'tqdm>=4.65.0',
]

# Web & API
web_packages = [
    'uvicorn>=0.23.0',
    'fastapi>=0.100.0',
    'pydantic>=2.0.0',
]

# Additional ML tools
ml_packages = [
    'pillow>=9.5.0',
    'opencv-python>=4.8.0',
    'librosa>=0.10.0',
    'soundfile>=0.12.0',
]

all_packages = core_packages + data_packages + hf_packages + data_source_packages + viz_packages + web_packages + ml_packages

failed_packages = []
installed_packages = []

for i, package in enumerate(all_packages, 1):
    try:
        print(f"\\n[{i}/{len(all_packages)}] Installing {package}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', package],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
            installed_packages.append(package)
        else:
            print(f"⚠️  {package} installation had issues: {result.stderr[:100]}")
            failed_packages.append(package)
    except subprocess.TimeoutExpired:
        print(f"⏱️  {package} installation timed out")
        failed_packages.append(package)
    except Exception as e:
        print(f"❌ {package} installation failed: {str(e)[:100]}")
        failed_packages.append(package)
    
    time.sleep(0.5)

print("\\n" + "=" * 70)
print("📊 INSTALLATION SUMMARY")
print("=" * 70)
print(f"✅ Installed: {len(installed_packages)}/{len(all_packages)}")
print(f"❌ Failed: {len(failed_packages)}/{len(all_packages)}")

if failed_packages:
    print(f"\\nFailed packages: {', '.join(failed_packages)}")

# Verify PyTorch installation
print("\\n" + "=" * 70)
print("🔍 VERIFYING PYTORCH INSTALLATION")
print("=" * 70)

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

# Verify HuggingFace
try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

# Verify datasets
try:
    import datasets
    print(f"✅ Datasets version: {datasets.__version__}")
except ImportError as e:
    print(f"❌ Datasets import failed: {e}")

# Verify Kaggle
try:
    import kaggle
    print(f"✅ Kaggle API available")
except ImportError as e:
    print(f"⚠️  Kaggle import failed: {e}")

print("\\n" + "=" * 70)
print("🎉 DEPENDENCY INSTALLATION COMPLETE")
print("=" * 70)
`);

    console.log('[create-pytorch-sandbox] Installation output:', installResult.logs?.stdout || installResult);

    if (installResult.error) {
      console.error('[create-pytorch-sandbox] Installation error:', installResult.error);
      // Don't fail on installation errors - some packages might have warnings
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
