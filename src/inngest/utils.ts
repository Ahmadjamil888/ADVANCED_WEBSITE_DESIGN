import { Sandbox } from "@e2b/code-interpreter";
import { SANDBOX_TIMEOUT } from "./types";

let sandboxInstance: Sandbox | null = null;

export async function getSandbox(sandboxId?: string) {
  if (sandboxInstance) {
    return sandboxInstance;
  }

  const sandbox = sandboxId 
    ? await Sandbox.connect(sandboxId)
    : await Sandbox.create("python3");
  
  await sandbox.setTimeout(SANDBOX_TIMEOUT);
  
  // Install common ML dependencies
  await sandbox.commands.run("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu");
  await sandbox.commands.run("pip install transformers datasets evaluate scikit-learn");
  
  sandboxInstance = sandbox;
  return sandbox;
}

export async function saveModelArtifacts(sandbox: Sandbox, modelPath: string, metadata: any) {
  // Save model artifacts to a persistent storage
  const modelData = await sandbox.files.read(modelPath);
  const artifacts = {
    modelData,
    metadata: {
      ...metadata,
      savedAt: new Date().toISOString(),
    },
  };
  
  // In a real implementation, you would save this to a database or cloud storage
  return artifacts;
}

export function generateModelCode(config: any): string {
  // This is a simplified example - in reality, you'd generate more complex model architectures
  return `
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import json
import os

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

def train_model():
    # Model configuration
    input_size = ${JSON.stringify(config.inputShape[0])}
    output_size = ${JSON.stringify(config.outputShape[0])}
    epochs = ${config.epochs || 10}
    batch_size = ${config.batchSize || 32}
    learning_rate = ${config.learningRate || 0.001}
    
    # Initialize model, loss, and optimizer
    model = CustomModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Sample data - in a real scenario, load your dataset here
    # X_train, y_train, X_val, y_val = load_data()
    # train_dataset = CustomDataset(X_train, y_train)
    # val_dataset = CustomDataset(X_val, y_val)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/trained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': input_size,
            'output_size': output_size,
            'architecture': 'CustomModel'
        }
    }, model_path)
    
    return model_path

if __name__ == '__main__':
    model_path = train_model()
    print(f"Model saved to: {model_path}")
`;
}
`;
}
