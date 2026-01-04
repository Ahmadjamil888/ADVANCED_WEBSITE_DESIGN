# User Guide - AI Model Generator Platform

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Dashboard Overview](#dashboard-overview)
4. [Creating Your First Model](#creating-your-first-model)
5. [Custom Datasets and Models](#custom-datasets-and-models)
6. [Usage Tracking](#usage-tracking)
7. [Billing and Plans](#billing-and-plans)
8. [Using Your Deployed Model](#using-your-deployed-model)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Getting Started

### Sign Up

1. Visit https://zehanxtech.com
2. Click "Free Trial" button
3. Enter your email and password
4. Verify your email
5. You're ready to go!

### First Login

1. Go to https://zehanxtech.com/login
2. Enter your credentials
3. You'll be redirected to the AI Model Generator dashboard

---

## Authentication

### Login/Signup

The platform uses Supabase authentication with support for:
- Email/Password
- Google OAuth
- Apple OAuth

### Session Management

- Sessions are automatically managed
- You'll be logged out after 30 days of inactivity
- You can manually sign out from Settings

### API Authentication

To use the API programmatically:

1. Go to Settings
2. Copy your API Token
3. Use in requests: `Authorization: Bearer YOUR_TOKEN`

---

## Dashboard Overview

### Main Tabs

#### Generator Tab
- Create new AI models
- Upload custom datasets
- Upload custom models
- Monitor generation progress
- View deployment results

#### Usage Tab
- Track tokens used
- View APIs created
- Monitor models deployed
- See monthly requests
- Check monthly costs

#### Billing Tab
- View current plan
- See available plans
- Upgrade/downgrade plans
- View billing history

#### Settings Tab
- Update profile
- Manage API keys
- Configure webhooks
- Set preferences

---

## Creating Your First Model

### Step 1: Navigate to Generator

Click the "Generator" tab in the dashboard.

### Step 2: Write Your Prompt

In the "Model Description" field, describe what you want:

**Examples:**
- "Create a sentiment analysis model that classifies text as positive, negative, or neutral"
- "Build a neural network for image classification"
- "Generate a recommendation system for e-commerce"

### Step 3: (Optional) Upload Custom Dataset

Click "Upload Custom Dataset" and select a file:
- Supported formats: CSV, JSON, XLSX
- Maximum size: 100MB
- Used for training your model

### Step 4: (Optional) Upload Custom Model

Click "Upload Custom Model" and select a file:
- Supported formats: .pth, .h5, .pb, .onnx, .safetensors
- Maximum size: 500MB
- Used as base model for fine-tuning

### Step 5: Generate

Click "Generate Model" button and wait for completion.

### Generation Process

The system will:
1. Generate code using AI (Groq, Gemini, or DeepSeek)
2. Create an E2B sandbox
3. Install dependencies
4. Train the model
5. Deploy to port 49999

Each step shows real-time progress.

### After Generation

Once complete, you'll see:
- Deployment URL
- Sandbox ID
- "Visit Model" button to test

---

## Custom Datasets and Models

### Uploading Datasets

**Supported Formats:**
- CSV: Comma-separated values
- JSON: JSON array or objects
- XLSX: Excel spreadsheets

**Best Practices:**
- Include headers/column names
- Clean data (remove nulls if possible)
- Balanced classes for classification
- Reasonable file size (< 100MB)

**Example CSV:**
```csv
text,sentiment
"Great product!",positive
"Terrible experience",negative
"It's okay",neutral
```

### Uploading Models

**Supported Formats:**
- PyTorch: .pth files
- TensorFlow: .h5 files
- ONNX: .onnx files
- Hugging Face: .safetensors files

**Requirements:**
- Model must be compatible with Python
- Include any necessary configuration files
- Document input/output shapes

---

## Usage Tracking

### Understanding Metrics

**Tokens Used**
- Counted per API call
- Based on input/output size
- Varies by model

**APIs Created**
- Number of deployed models
- Counts towards plan limit

**Models Deployed**
- Successfully deployed models
- Includes all versions

**Requests This Month**
- Total API calls made
- Resets monthly

**Cost This Month**
- Calculated based on usage
- Varies by plan tier

### Viewing Detailed Usage

1. Click "Usage" tab
2. See all metrics
3. Export data (coming soon)

---

## Billing and Plans

### Plan Comparison

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Price | $0 | $80/mo | $100/mo |
| AI Models | 1 | 10 | Unlimited |
| API Calls | 1,000/mo | 100,000/mo | Unlimited |
| Support | Basic | Priority | 24/7 |
| Custom Models | No | Yes | Yes |
| Webhooks | No | Yes | Yes |

### Upgrading Your Plan

1. Click "Billing" tab
2. Select desired plan
3. Click "Upgrade"
4. Complete payment
5. Instant activation

### Downgrading

1. Click "Billing" tab
2. Click "Downgrade"
3. Changes take effect next billing cycle

### Billing Cycle

- Monthly billing on the same date each month
- Automatic renewal
- Cancel anytime

---

## Using Your Deployed Model

### Model Endpoints

Your deployed model has three main endpoints:

#### 1. Health Check
```bash
GET /health
```

Returns model status and uptime.

#### 2. Get Model Info
```bash
GET /info
```

Returns model type, framework, and endpoints.

#### 3. Make Predictions
```bash
POST /predict
Content-Type: application/json

{"input": [...]}
```

Returns predictions and confidence scores.

### Integration Examples

#### Python
```python
import requests

url = "https://your-sandbox-url.e2b.dev/predict"
data = {"input": [1, 2, 3, 4]}

response = requests.post(url, json=data)
result = response.json()
print(result['prediction'])
```

#### JavaScript
```javascript
const url = "https://your-sandbox-url.e2b.dev/predict";
const data = { input: [1, 2, 3, 4] };

const response = await fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

const result = await response.json();
console.log(result.prediction);
```

#### cURL
```bash
curl -X POST https://your-sandbox-url.e2b.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3, 4]}'
```

---

## Troubleshooting

### Model Generation Failed

**Problem**: Generation fails with error message

**Solutions**:
1. Check your prompt is clear and specific
2. Ensure dataset format is correct
3. Try a simpler model first
4. Check API quota

### Model Not Responding

**Problem**: Deployment URL returns 502/503

**Solutions**:
1. Wait 30 seconds after deployment
2. Check model health: `GET /health`
3. Verify URL is correct
4. Check firewall settings

### High Token Usage

**Problem**: Using more tokens than expected

**Solutions**:
1. Reduce input size
2. Use simpler models
3. Batch requests
4. Upgrade to Pro plan

### Quota Exceeded

**Problem**: "Insufficient quota" error

**Solutions**:
1. Upgrade your plan
2. Wait for monthly reset
3. Delete unused models
4. Contact support

---

## FAQ

### Q: How long does model generation take?

A: Typically 2-5 minutes depending on model complexity.

### Q: Can I use my own data?

A: Yes! Upload CSV, JSON, or XLSX files as custom datasets.

### Q: What models can I create?

A: Any Python-based model (PyTorch, TensorFlow, scikit-learn, etc.)

### Q: Is my data secure?

A: Yes, data is encrypted and stored securely. Models run in isolated E2B sandboxes.

### Q: Can I download my model?

A: Yes, from the deployment URL or via API.

### Q: What's the model backend port?

A: Port 49999 - the standard E2B sandbox model backend port.

### Q: Can I integrate with my app?

A: Yes! Use the REST API endpoints to integrate into any application.

### Q: What if I exceed my quota?

A: Upgrade your plan or wait for monthly reset.

### Q: How do I cancel my subscription?

A: Go to Billing tab and click "Cancel Subscription".

### Q: Do you offer refunds?

A: Yes, within 14 days of purchase.

### Q: Can I use the API?

A: Yes! All plans include API access. Get your token in Settings.

### Q: What's included in support?

A: Free: Community support, Pro: Email support, Enterprise: 24/7 phone support.

### Q: Can I create multiple models?

A: Yes! Free plan: 1, Pro: 10, Enterprise: Unlimited.

### Q: How are costs calculated?

A: Based on tokens used and API calls made. See Usage tab for breakdown.

---

## Getting Help

### Support Channels

- **Email**: support@zehanxtech.com
- **Chat**: In-app chat (Pro/Enterprise)
- **Documentation**: https://docs.zehanxtech.com
- **Status**: https://status.zehanxtech.com

### Reporting Issues

1. Go to Settings
2. Click "Report Issue"
3. Describe the problem
4. Attach screenshots if helpful
5. Submit

---

## Best Practices

### Model Creation

- Use clear, specific prompts
- Provide quality training data
- Start with simple models
- Test before production

### API Usage

- Implement error handling
- Use rate limiting
- Cache responses when possible
- Monitor usage regularly

### Security

- Keep API tokens secret
- Use HTTPS for all requests
- Rotate tokens periodically
- Enable webhooks for monitoring

---

**Last Updated**: November 16, 2025
**Version**: 1.0.0
