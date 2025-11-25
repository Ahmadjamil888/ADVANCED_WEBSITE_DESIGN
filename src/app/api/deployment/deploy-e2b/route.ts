import { NextRequest, NextResponse } from 'next/server';

declare global {
  var activePyTorchSandbox: any;
  var deploymentUrls: Map<string, string>;
}

if (!global.deploymentUrls) {
  global.deploymentUrls = new Map();
}

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const { sandboxId, modelPath, appCode, modelType = 'custom' } = await request.json();

    if (!sandboxId) {
      return NextResponse.json(
        { error: 'Sandbox ID is required' },
        { status: 400 }
      );
    }

    if (!process.env.E2B_API_KEY) {
      return NextResponse.json(
        { error: 'E2B_API_KEY is not configured' },
        { status: 500 }
      );
    }

    console.log('[deploy-e2b] Starting E2B deployment...');
    console.log('[deploy-e2b] Sandbox ID:', sandboxId);
    console.log('[deploy-e2b] Model path:', modelPath);
    console.log('[deploy-e2b] Model type:', modelType);

    // Import E2B SDK
    let Sandbox: any;
    try {
      const e2bModule: any = await import('@e2b/code-interpreter');
      Sandbox = e2bModule.Sandbox || e2bModule.default;
      
      if (!Sandbox) {
        throw new Error('Sandbox class not found');
      }
    } catch (importError) {
      console.error('[deploy-e2b] Import error:', importError);
      throw new Error(`Failed to import E2B SDK: ${importError instanceof Error ? importError.message : 'Unknown error'}`);
    }

    let sandbox: any = global.activePyTorchSandbox;

    // If no active sandbox, connect to existing one
    if (!sandbox) {
      console.log('[deploy-e2b] Connecting to existing sandbox...');
      sandbox = await Sandbox.connect(sandboxId, {
        apiKey: process.env.E2B_API_KEY,
      });
      global.activePyTorchSandbox = sandbox;
    }

    // Create Flask/FastAPI app for serving the model
    const deploymentAppCode = appCode || `
import torch
import json
from flask import Flask, request, jsonify
from pathlib import Path
import traceback

app = Flask(__name__)

# Load model
try:
    model_path = '${modelPath}'
    model = torch.load(model_path)
    model.eval()
    print(f'Model loaded from {model_path}')
except Exception as e:
    print(f'Error loading model: {e}')
    model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Convert input to tensor
        input_tensor = torch.tensor(data.get('input', []), dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to JSON-serializable format
        result = output.cpu().numpy().tolist() if hasattr(output, 'cpu') else output.tolist()
        
        return jsonify({
            'success': True,
            'prediction': result,
            'model_type': '${modelType}'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model_type': '${modelType}',
        'model_loaded': model is not None,
        'endpoints': ['/health', '/predict', '/info']
    })

if __name__ == '__main__':
    # Port 49999 is the E2B sandbox model backend port
    app.run(host='0.0.0.0', port=49999, debug=False)
`;

    // Write the deployment app
    console.log('[deploy-e2b] Writing deployment app...');
    const writeCode = `
import os
with open('app.py', 'w') as f:
    f.write("""${deploymentAppCode.replace(/"/g, '\\"')}""")
print('app.py written successfully')
`;
    const writeResult = await sandbox.runCode(writeCode);
    console.log('[deploy-e2b] Write result:', writeResult.logs?.stdout || writeResult);

    // Install Flask
    console.log('[deploy-e2b] Installing Flask...');
    const flaskInstall = await sandbox.runCode(`
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'flask'])
print('Flask installed')
`);

    console.log('[deploy-e2b] Flask installation output:', flaskInstall.logs?.stdout || flaskInstall);

    // Start the Flask server in background with nohup
    console.log('[deploy-e2b] Starting Flask server...');
    const serverStart = await sandbox.runCode(`
import subprocess
import time
import sys
import os

# Start Flask server in background with nohup
with open('server.log', 'w') as log_file:
    process = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,  # Detach from parent process
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )

# Wait for server to start
time.sleep(5)

# Check if process is still running
if process.poll() is None:
    print(f'Flask server started successfully with PID {process.pid}')
else:
    with open('server.log', 'r') as f:
        print(f'Server failed to start: {f.read()}')
`);

    console.log('[deploy-e2b] Server start output:', serverStart.logs?.stdout || serverStart);

    // Get sandbox URL using proper E2B method
    // Port 49999 is the E2B sandbox model backend port for AI model serving
    const sandboxHost = sandbox.getHost(49999);
    const sandboxUrl = `https://${sandboxHost}`;

    // Store deployment URL
    global.deploymentUrls.set(sandboxId, sandboxUrl);

    console.log('[deploy-e2b] Deployment URL:', sandboxUrl);

    // Verify deployment by checking if Flask process is running
    console.log('[deploy-e2b] Verifying deployment...');
    const verifyCode = `
import subprocess
import time

# Check if Flask process is running
for i in range(5):
    try:
        result = subprocess.run(['pgrep', '-f', 'app.py'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f'Flask server is running with PID: {result.stdout.strip()}')
            break
    except Exception as e:
        print(f'Attempt {i+1}: Checking process - {e}')
        time.sleep(1)
`;

    const verifyResult = await sandbox.runCode(verifyCode);
    console.log('[deploy-e2b] Verification output:', verifyResult.logs?.stdout || verifyResult);

    return NextResponse.json({
      success: true,
      sandboxId,
      deploymentUrl: sandboxUrl,
      modelPath,
      modelType,
      endpoints: {
        health: `${sandboxUrl}/health`,
        predict: `${sandboxUrl}/predict`,
        info: `${sandboxUrl}/info`,
      },
      message: 'Model deployed successfully on E2B',
    });
  } catch (error) {
    console.error('[deploy-e2b] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to deploy model',
      },
      { status: 500 }
    );
  }
}
