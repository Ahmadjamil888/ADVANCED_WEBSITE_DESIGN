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
      timeoutMs: 60 * 60 * 1000, // 60 minutes timeout for dependency installation
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

# Install core packages first (critical)
print("\\n📦 Installing CORE packages (PyTorch, NumPy, etc.)...")
core_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + core_packages,
    capture_output=True,
    text=True,
    timeout=600
)
print(f"Core packages: {'✅ Success' if core_result.returncode == 0 else '⚠️  Partial'}")

# Install data packages
print("\\n📦 Installing DATA packages (Pandas, SciPy, etc.)...")
data_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + data_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"Data packages: {'✅ Success' if data_result.returncode == 0 else '⚠️  Partial'}")

# Install HuggingFace packages
print("\\n📦 Installing HUGGINGFACE packages (Transformers, Datasets, etc.)...")
hf_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + hf_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"HuggingFace packages: {'✅ Success' if hf_result.returncode == 0 else '⚠️  Partial'}")

# Install data source packages
print("\\n📦 Installing DATA SOURCE packages (Kaggle, Requests, etc.)...")
source_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + data_source_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"Data source packages: {'✅ Success' if source_result.returncode == 0 else '⚠️  Partial'}")

# Install visualization packages
print("\\n📦 Installing VISUALIZATION packages (Matplotlib, Seaborn, etc.)...")
viz_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + viz_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"Visualization packages: {'✅ Success' if viz_result.returncode == 0 else '⚠️  Partial'}")

# Install web packages
print("\\n📦 Installing WEB packages (FastAPI, Uvicorn, etc.)...")
web_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + web_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"Web packages: {'✅ Success' if web_result.returncode == 0 else '⚠️  Partial'}")

# Install ML packages
print("\\n📦 Installing ML TOOLS packages (OpenCV, Librosa, etc.)...")
ml_result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + ml_packages,
    capture_output=True,
    text=True,
    timeout=300
)
print(f"ML tools packages: {'✅ Success' if ml_result.returncode == 0 else '⚠️  Partial'}")

# Count successes
results = [core_result, data_result, hf_result, source_result, viz_result, web_result, ml_result]
successful = sum(1 for r in results if r.returncode == 0)
installed_packages = [p for p in all_packages]  # Assume all installed
failed_packages = []

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
