# E2B + Git CLI Setup for HuggingFace Spaces Deployment

This guide shows how to set up E2B sandbox integration with Git CLI for reliable HuggingFace Spaces deployment.

## 🚀 Quick Setup

### 1. Install E2B SDK

```bash
npm install @e2b/sdk
```

### 2. Get E2B API Key

1. Go to [E2B Dashboard](https://e2b.dev/dashboard)
2. Create account and get API key
3. Add to your `.env.local`:

```env
E2B_API_KEY=your_e2b_api_key_here
HF_ACCESS_TOKEN=your_huggingface_token_here
```

### 3. Update Package.json

Add E2B dependency:

```json
{
  "dependencies": {
    "@e2b/sdk": "^0.16.0"
  }
}
```

## 🔧 Implementation Steps

### Step 1: Replace Simulated E2B Code

In `src/lib/e2b-git-cli.ts`, replace the simulated code with real E2B SDK:

```typescript
import { Sandbox } from '@e2b/sdk'

// Replace this:
const sandboxId = `e2b-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

// With this:
const sandbox = await Sandbox.create('python3')
const sandboxId = sandbox.id
```

### Step 2: Implement Real Command Execution

Replace the simulated `executeCommandsInSandbox` with:

```typescript
private async executeCommandsInSandbox(sandboxId: string, commands: string[]): Promise<any> {
  const sandbox = await Sandbox.connect(sandboxId)
  
  for (const command of commands) {
    const result = await sandbox.process.start({
      cmd: command,
      timeout: this.config.timeout
    })
    
    if (result.exitCode !== 0) {
      throw new Error(`Command failed: ${command}\nError: ${result.stderr}`)
    }
    
    console.log(`✅ ${command}`)
    console.log(result.stdout)
  }
  
  return { success: true, output: 'Commands executed successfully' }
}
```

### Step 3: Update Environment Variables

Add to your `.env.local`:

```env
# E2B Configuration
E2B_API_KEY=your_e2b_api_key_here

# HuggingFace Configuration  
HF_ACCESS_TOKEN=your_huggingface_token_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Git Configuration
GIT_USER_EMAIL=ai@zehanxtech.com
GIT_USER_NAME=zehanx AI
```

### Step 4: Update Deployment Routes

In `src/app/api/ai-workspace/deploy-e2b-cli/route.ts`, replace simulated functions with real E2B calls:

```typescript
import { E2BGitCLIDeployer } from '@/lib/e2b-git-cli'

async function deployWithE2BGitCLI(spaceName: string, hfToken: string, modelType: string, options: DeploymentOptions = {}) {
  const e2bApiKey = process.env.E2B_API_KEY
  if (!e2bApiKey) {
    throw new Error('E2B_API_KEY not configured')
  }
  
  const deployer = new E2BGitCLIDeployer(e2bApiKey, hfToken)
  const files = generateMLModelFiles(modelType, options.prompt || '')
  
  return await deployer.deployToHuggingFaceSpace(spaceName, 'Ahmadjamil888', files)
}
```

## 🔄 Complete Git CLI Workflow

The E2B + Git CLI deployment follows this exact workflow:

### 1. E2B Sandbox Setup
```bash
# Install system dependencies
apt-get update && apt-get install -y git curl

# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

# Install HuggingFace CLI
pip install huggingface_hub

# Configure Git
git config --global user.email "ai@zehanxtech.com"
git config --global user.name "zehanx AI"

# Login to HuggingFace
echo "YOUR_HF_TOKEN" | huggingface-cli login --token
```

### 2. Clone HuggingFace Space
```bash
cd /workspace
git clone https://oauth2:YOUR_HF_TOKEN@huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME
```

### 3. Generate and Add Files
```bash
# Create all ML pipeline files
echo "content" > app.py
echo "content" > requirements.txt
echo "content" > README.md
echo "content" > train.py
echo "content" > inference.py
echo "content" > config.py
echo "content" > model.py
echo "content" > utils.py
echo "content" > dataset.py
echo "content" > Dockerfile
```

### 4. Git Commit and Push
```bash
# Stage all files
git add .

# Check status
git status

# Commit with descriptive message
git commit -m "Add complete AI model files (10 files) - zehanx tech E2B + Git CLI deployment"

# Push to HuggingFace
git push origin main

# Verify
git log --oneline -n 3
```

## 🎯 Benefits of E2B + Git CLI

### ✅ Advantages
- **Guaranteed File Upload**: All files are pushed via Git
- **Proper Git History**: Clean commit messages and history
- **No File Loss**: E2B ensures reliable execution
- **Official Tools**: Uses HuggingFace CLI and Git
- **Isolated Environment**: E2B sandbox prevents conflicts
- **Complete Pipeline**: All 10 ML files deployed

### 🔧 Reliability Features
- **Retry Logic**: Automatic retries on failures
- **Error Handling**: Comprehensive error reporting
- **Logging**: Detailed execution logs
- **Verification**: Post-deployment verification
- **Cleanup**: Automatic sandbox cleanup

## 📋 File Structure Deployed

The E2B + Git CLI deployment creates this complete ML pipeline:

```
HuggingFace Space/
├── app.py              # Advanced Gradio interface
├── requirements.txt    # All dependencies
├── README.md          # HuggingFace Space metadata
├── train.py           # Training pipeline
├── inference.py       # Inference utilities
├── config.py          # Configuration management
├── model.py           # Model architecture
├── utils.py           # Utility functions
├── dataset.py         # Data handling
└── Dockerfile         # Container configuration
```

## 🚨 Troubleshooting

### Common Issues

1. **E2B API Key Missing**
   ```
   Error: E2B_API_KEY not configured
   ```
   **Solution**: Add E2B API key to `.env.local`

2. **HuggingFace Token Invalid**
   ```
   Error: Failed to authenticate with HuggingFace
   ```
   **Solution**: Check HF token has write permissions

3. **Git Clone Fails**
   ```
   Error: Failed to clone HF Space
   ```
   **Solution**: Ensure space exists and token has access

4. **File Upload Fails**
   ```
   Error: Git push failed
   ```
   **Solution**: Check file content and Git configuration

### Debug Mode

Enable debug logging:

```typescript
const deployer = new E2BGitCLIDeployer(e2bApiKey, hfToken)
deployer.setDebugMode(true) // Add this method for detailed logs
```

## 🎉 Success Verification

After deployment, verify:

1. **Space URL**: `https://huggingface.co/spaces/USERNAME/SPACE_NAME`
2. **All Files Present**: Check all 10 files are uploaded
3. **Gradio Interface**: App loads and works correctly
4. **Git History**: Clean commit history visible

## 📞 Support

If you encounter issues:

1. Check E2B dashboard for sandbox logs
2. Verify HuggingFace token permissions
3. Review Git configuration in E2B
4. Check file content encoding

---

**🚀 Powered by zehanx tech - E2B + Git CLI Deployment**