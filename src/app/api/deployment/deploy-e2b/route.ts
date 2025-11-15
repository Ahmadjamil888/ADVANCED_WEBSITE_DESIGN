import { NextRequest, NextResponse } from 'next/server';
import { Sandbox } from '@e2b/code-interpreter';

declare global {
  var activePyTorchSandbox: any;
  var deploymentUrls: Map<string, string>;
}

if (!global.deploymentUrls) {
  global.deploymentUrls = new Map();
}

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

    let sandbox = global.activePyTorchSandbox;

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
    app.run(host='0.0.0.0', port=8000, debug=False)
`;

    // Write the deployment app
    console.log('[deploy-e2b] Writing deployment app...');
    await sandbox.writeFile('app.py', deploymentAppCode);

    // Install Flask
    console.log('[deploy-e2b] Installing Flask...');
    const flaskInstall = await sandbox.runPython(`
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'flask'])
print('Flask installed')
`);

    console.log('[deploy-e2b] Flask installation output:', flaskInstall.stdout);

    // Start the Flask server in background
    console.log('[deploy-e2b] Starting Flask server...');
    const serverStart = await sandbox.runPython(`
import subprocess
import time
import sys

# Start Flask server in background
process = subprocess.Popen(
    [sys.executable, 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for server to start
time.sleep(3)

# Check if process is still running
if process.poll() is None:
    print('Flask server started successfully')
else:
    stdout, stderr = process.communicate()
    print(f'Server failed to start: {stderr}')
`);

    console.log('[deploy-e2b] Server start output:', serverStart.stdout);

    // Get sandbox URL
    const sandboxUrl = `https://${sandboxId}.e2b.dev:8000`;

    // Store deployment URL
    global.deploymentUrls.set(sandboxId, sandboxUrl);

    console.log('[deploy-e2b] Deployment URL:', sandboxUrl);

    // Verify deployment
    console.log('[deploy-e2b] Verifying deployment...');
    const verifyCode = `
import requests
import time

for i in range(5):
    try:
        response = requests.get('http://localhost:8000/health', timeout=2)
        if response.status_code == 200:
            print(f'�u2713 Server is healthy: {response.json()}')
            break
    except Exception as e:
        print(f'Attempt {i+1}: Server not ready - {e}')
        time.sleep(1)
`;

    const verifyResult = await sandbox.runPython(verifyCode);
    console.log('[deploy-e2b] Verification output:', verifyResult.stdout);

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
