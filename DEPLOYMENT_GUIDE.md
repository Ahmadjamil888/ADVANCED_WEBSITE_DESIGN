# Deployment Guide

## Vercel Deployment Fix

This project had a serverless function size issue due to heavy AI/ML dependencies. Here's what was fixed:

### Problem
- Vercel was trying to bundle Python AI service dependencies
- `onnxruntime-node` (404.69 MB) and `@huggingface/transformers` (35.71 MB) exceeded the 250 MB limit

### Solution Applied

1. **Updated AI Model**: Switched from DeepSeek-R1 to DistilGPT-2 (lighter model)
   - File: `python-ai-service/main.py`
   - Model: `distilbert/distilgpt2` (much smaller footprint)

2. **Added .vercelignore**: Excluded Python service from deployment
   - Ignores `python-ai-service/` directory
   - Ignores all Python files and dependencies

3. **Updated Next.js Config**: Added webpack externals
   - Excludes heavy AI/ML packages from bundling
   - Ignores Python files in watch mode

4. **Cleaned Dependencies**: Reduced Python requirements
   - Removed heavy packages like `accelerate`, `sentencepiece`
   - Kept only essential packages

### Current Architecture

- **Frontend**: Next.js app with TypeScript API routes
- **AI Service**: Separate Python service (not deployed to Vercel)
- **Chat API**: Uses rule-based responses in `/api/chat/route.ts`

### Deployment Commands

```bash
# Build and deploy
npm run build
vercel --prod

# Or use Vercel CLI
vercel deploy --prod
```

### Alternative AI Service Deployment

If you want to use the Python AI service:

1. Deploy it separately (Railway, Render, or AWS Lambda)
2. Update the chat API to call the external service
3. Set environment variables for the service URL

### Bundle Size Monitoring

To check bundle sizes:
```bash
npm run build
# Check .next/server/pages/ for function sizes
```

The current setup should deploy successfully within Vercel's limits.