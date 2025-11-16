import { NextRequest, NextResponse } from 'next/server';

declare global {
  var activeSandbox: any;
  var sandboxId: string;
}

export const maxDuration = 300;

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

    console.log('[train-model] Starting lightweight model training...');
    console.log('[train-model] Sandbox ID:', sandboxId);

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

    let sandbox: any = global.activeSandbox;

    // If no active sandbox or different sandbox ID, create/connect to sandbox
    if (!sandbox || global.sandboxId !== sandboxId) {
      console.log('[train-model] Creating new sandbox for training...');
      sandbox = await Sandbox.create({
        apiKey: process.env.E2B_API_KEY,
        timeoutMs: 60 * 60 * 1000, // 1 hour timeout for training
      });
      global.activeSandbox = sandbox;
      global.sandboxId = sandbox.sandboxId;
    }

    // Use PyTorch training code
    const pytorchCode = `
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("🚀 PYTORCH MODEL TRAINING")
print("=" * 70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate dataset
print("\\nGenerating synthetic dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size=20):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\\nStarting training...")
print("-" * 70)

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_correct = (val_predicted == y_test_tensor).sum().item()
        val_accuracy = val_correct / len(y_test_tensor)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    print(f"Epoch {epoch}/10 - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} - Val Loss: {val_loss.item():.4f} - Val Accuracy: {val_accuracy:.4f}")

print("-" * 70)

# Save model
print("\\nSaving model...")
torch.save(model.state_dict(), '/tmp/model.pt')
print("✅ Model saved to /tmp/model.pt")

print("\\n" + "=" * 70)
print("🎉 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
`;

    // Run the PyTorch training code
    console.log('[train-model] Running PyTorch training code...');
    const result = await sandbox.runCode(pytorchCode);

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
        if filename.endswith(('.pkl', '.joblib', '.pt', '.pth')):
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
