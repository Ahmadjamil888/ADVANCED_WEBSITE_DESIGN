# E2B Model Backend Port Configuration Guide

## Overview

This guide explains the port configuration for AI models generated and deployed through the E2B sandbox infrastructure. The system uses **port 49999** as the standard model backend port.

## Port 49999: E2B Sandbox Model Backend Port

### What is Port 49999?

**Port 49999** is the dedicated port used by the E2B sandbox environment for serving your generated AI models. This is where your trained models will listen for inference requests.

### Why Port 49999?

- **Standard E2B Port**: Port 49999 is the conventional port used by E2B for model serving
- **Isolation**: Keeps model traffic separate from other services
- **Consistency**: Ensures all generated models use the same port for predictable deployment
- **Firewall Friendly**: High port number (>1024) doesn't require root privileges

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│                   (Next.js Dashboard)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                    HTTP/HTTPS
                         │
        ┌────────────────▼────────────────┐
        │   E2B Sandbox Environment       │
        │                                 │
        │  ┌──────────────────────────┐   │
        │  │  FastAPI/Flask Server    │   │
        │  │  (Your AI Model)         │   │
        │  │  Listening on :49999     │   │
        │  └──────────────────────────┘   │
        │                                 │
        └─────────────────────────────────┘
```

## How Models Use Port 49999

### 1. Model Deployment

When you generate and deploy an AI model:

```typescript
// In src/lib/e2b.ts
const deploymentUrl = await manager.deployAPI(
  '/home/user/app.py',
  49999  // ← Model backend port
);
```

### 2. Model Endpoints

Your deployed model will be accessible at:

- **Health Check**: `https://{sandbox-host}/health`
- **Predictions**: `https://{sandbox-host}/predict`
- **Model Info**: `https://{sandbox-host}/info`

### 3. Making Predictions

```typescript
// Example: Calling your deployed model
const response = await fetch(`${deploymentUrl}/predict`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: [1, 2, 3, 4],  // Your input data
  }),
});

const prediction = await response.json();
console.log('Model prediction:', prediction);
```

## Configuration Files

### Files Using Port 49999

1. **`src/lib/e2b.ts`**
   - Default port for `getHost()` method
   - Default port for `deployAPI()` method
   - Contains documentation about the port

2. **`src/app/api/deploy/e2b/route.ts`**
   - Deployment endpoint that uses port 49999
   - Starts FastAPI server on port 49999

3. **`src/app/api/deployment/deploy-e2b/route.ts`**
   - Alternative deployment route
   - Flask server configuration on port 49999

## Environment Variables

Currently, port 49999 is hardcoded as the standard. If you need to customize it, you can:

1. Add to `.env.local`:
```env
E2B_MODEL_PORT=49999
```

2. Update the code to use the environment variable:
```typescript
const port = Number(process.env.E2B_MODEL_PORT ?? 49999);
```

## API Response Format

### Health Check Response
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction Response
```json
{
  "success": true,
  "prediction": [0.95, 0.03, 0.02],
  "model_type": "custom",
  "confidence": 0.95
}
```

### Model Info Response
```json
{
  "model_type": "custom",
  "model_loaded": true,
  "endpoints": ["/health", "/predict", "/info"]
}
```

## Troubleshooting

### Model Not Responding on Port 49999

1. **Check Sandbox Status**
   ```bash
   # Verify sandbox is running
   curl https://{sandbox-host}/health
   ```

2. **Check Logs**
   - Look for deployment logs in the browser console
   - Check server logs in the E2B sandbox

3. **Verify Port Binding**
   - Ensure no other service is using port 49999
   - Check firewall rules

### Connection Refused

- Model server may still be starting (wait 30 seconds)
- Check if dependencies are installed correctly
- Verify the FastAPI/Flask app is properly configured

## Best Practices

1. **Always Use HTTPS**: E2B provides HTTPS URLs for secure communication
2. **Handle Timeouts**: Set appropriate timeout values for long-running predictions
3. **Batch Requests**: For better performance, batch multiple predictions
4. **Monitor Resources**: Watch CPU/memory usage during model inference
5. **Cache Results**: Cache model predictions when possible to reduce latency

## Example: Complete Model Deployment Flow

```typescript
import { E2BManager } from '@/lib/e2b';

async function deployModel() {
  const manager = new E2BManager();
  
  // 1. Create sandbox
  await manager.createSandbox();
  
  // 2. Write model files
  await manager.writeFiles({
    'model.pt': modelData,
    'requirements.txt': 'torch==2.1.0\nfastapi==0.104.0\nuvicorn==0.24.0\n',
    'app.py': fastApiCode,
  });
  
  // 3. Install dependencies
  await manager.installDependencies();
  
  // 4. Deploy on port 49999
  const deploymentUrl = await manager.deployAPI(
    '/home/user/app.py',
    49999
  );
  
  // 5. Make predictions
  const response = await fetch(`${deploymentUrl}/predict`, {
    method: 'POST',
    body: JSON.stringify({ input: [1, 2, 3] }),
  });
  
  const result = await response.json();
  console.log('Prediction:', result);
  
  // 6. Cleanup
  await manager.close();
}
```

## Summary

- **Port 49999** is the standard E2B sandbox model backend port
- All generated AI models listen on this port
- Use HTTPS for secure communication
- The port is configured in `src/lib/e2b.ts` and deployment routes
- Models provide `/health`, `/predict`, and `/info` endpoints

For more information, see the [E2B Documentation](https://e2b.dev/docs).
