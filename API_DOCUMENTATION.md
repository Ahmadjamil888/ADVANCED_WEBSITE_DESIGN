# API Documentation - AI Model Generator

## Overview

This document provides comprehensive documentation for the AI Model Generator API, including endpoints, callbacks, URL structures, and backend integration.

---

## Table of Contents

1. [Base URL](#base-url)
2. [Authentication](#authentication)
3. [Model Generation API](#model-generation-api)
4. [Usage Tracking API](#usage-tracking-api)
5. [Deployment API](#deployment-api)
6. [Callbacks](#callbacks)
7. [Response Formats](#response-formats)
8. [Error Handling](#error-handling)
9. [Backend Integration](#backend-integration)
10. [Examples](#examples)

---

## Base URL

```
Production: https://api.zehanxtech.com/api
Development: http://localhost:3000/api
```

---

## Authentication

All API requests require authentication via Bearer token in the Authorization header.

### Header Format
```
Authorization: Bearer <user_token>
```

### Getting Your Token

1. Sign up/Login at the dashboard
2. Navigate to Settings
3. Copy your API token
4. Use in all API requests

---

## Model Generation API

### Generate AI Model

**Endpoint**: `POST /ai/generate`

**Description**: Generates and deploys an AI model based on your prompt.

**Request**

```bash
curl -X POST https://api.zehanxtech.com/api/ai/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d {
    "prompt": "Create a sentiment analysis model",
    "userId": "user-123",
    "customDataset": "file",
    "customModel": "file"
  }
```

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| prompt | string | Yes | Description of the AI model to create |
| userId | string | Yes | Your user ID |
| customDataset | file | No | Custom dataset file (CSV, JSON, XLSX) |
| customModel | file | No | Custom model file (.pth, .h5, .pb, .onnx) |

**Response**

```json
{
  "success": true,
  "deploymentUrl": "https://sandbox-xyz.e2b.dev/",
  "sandboxId": "sandbox-123",
  "modelId": "model-456",
  "message": "Model generated and deployed successfully"
}
```

**Status Codes**

- `200`: Success
- `400`: Bad request
- `401`: Unauthorized
- `500`: Server error

---

## Usage Tracking API

### Get Usage Statistics

**Endpoint**: `GET /usage?userId={userId}`

**Description**: Retrieves usage statistics for the authenticated user.

**Request**

```bash
curl -X GET "https://api.zehanxtech.com/api/usage?userId=user-123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**

```json
{
  "tokensUsed": 125000,
  "apisCreated": 5,
  "modelsDeployed": 3,
  "requestsThisMonth": 1250,
  "costThisMonth": 45.50
}
```

### Log API Usage

**Endpoint**: `POST /usage/log`

**Description**: Logs API usage for tracking and billing.

**Request**

```bash
curl -X POST https://api.zehanxtech.com/api/usage/log \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d {
    "endpoint": "/predict",
    "method": "POST",
    "tokensUsed": 150,
    "responseTime": 245
  }
```

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| endpoint | string | Yes | API endpoint called |
| method | string | Yes | HTTP method (GET, POST, etc.) |
| tokensUsed | number | Yes | Tokens consumed |
| responseTime | number | Yes | Response time in milliseconds |

---

## Deployment API

### Deploy to E2B Sandbox

**Endpoint**: `POST /deploy/e2b`

**Description**: Deploys your model to E2B sandbox on port 49999.

**Request**

```bash
curl -X POST https://api.zehanxtech.com/api/deploy/e2b \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d {
    "modelCode": "...",
    "requirements": ["numpy", "torch"],
    "port": 49999
  }
```

**Response**

```json
{
  "success": true,
  "deploymentUrl": "https://sandbox-xyz.e2b.dev/",
  "sandboxId": "sandbox-123",
  "port": 49999
}
```

### Get Deployment Status

**Endpoint**: `GET /deploy/status/{sandboxId}`

**Description**: Checks the status of a deployed model.

**Request**

```bash
curl -X GET "https://api.zehanxtech.com/api/deploy/status/sandbox-123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**

```json
{
  "status": "running",
  "sandboxId": "sandbox-123",
  "deploymentUrl": "https://sandbox-xyz.e2b.dev/",
  "uptime": 3600,
  "health": "healthy"
}
```

---

## Callbacks

### Deployment Callback

When your model is deployed, a callback is sent to your specified webhook URL.

**Callback URL**: Set in Settings → Webhooks

**Callback Payload**

```json
{
  "event": "model.deployed",
  "timestamp": "2025-11-16T22:35:00Z",
  "data": {
    "modelId": "model-456",
    "deploymentUrl": "https://sandbox-xyz.e2b.dev/",
    "sandboxId": "sandbox-123",
    "status": "running"
  }
}
```

### Training Callback

Sent when model training completes.

**Callback Payload**

```json
{
  "event": "model.training_complete",
  "timestamp": "2025-11-16T22:35:00Z",
  "data": {
    "modelId": "model-456",
    "accuracy": 0.95,
    "loss": 0.12,
    "epochs": 10
  }
}
```

### Error Callback

Sent when an error occurs during generation or deployment.

**Callback Payload**

```json
{
  "event": "model.error",
  "timestamp": "2025-11-16T22:35:00Z",
  "data": {
    "modelId": "model-456",
    "error": "Insufficient memory",
    "step": "training"
  }
}
```

---

## Response Formats

### Model Prediction Response

When calling your deployed model's `/predict` endpoint:

**Request**

```bash
curl -X POST https://sandbox-xyz.e2b.dev/predict \
  -H "Content-Type: application/json" \
  -d {
    "input": [1, 2, 3, 4]
  }
```

**Response**

```json
{
  "success": true,
  "prediction": [0.95, 0.05],
  "confidence": 0.95,
  "processingTime": 245
}
```

### Model Info Response

**Request**

```bash
curl -X GET https://sandbox-xyz.e2b.dev/info
```

**Response**

```json
{
  "modelType": "neural_network",
  "framework": "pytorch",
  "version": "1.0.0",
  "inputShape": [1, 4],
  "outputShape": [1, 2],
  "endpoints": ["/health", "/predict", "/info"]
}
```

### Health Check Response

**Request**

```bash
curl -X GET https://sandbox-xyz.e2b.dev/health
```

**Response**

```json
{
  "status": "healthy",
  "modelLoaded": true,
  "uptime": 3600,
  "memoryUsage": "512MB"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional error details"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_PROMPT | 400 | Prompt is empty or invalid |
| INSUFFICIENT_QUOTA | 429 | Exceeded API quota |
| UNAUTHORIZED | 401 | Invalid or missing authentication |
| MODEL_GENERATION_FAILED | 500 | Model generation failed |
| DEPLOYMENT_FAILED | 500 | Deployment to E2B failed |
| SANDBOX_ERROR | 500 | E2B sandbox error |

---

## Backend Integration

### Using Your Model as a Backend

Once deployed, your model becomes a REST API that you can integrate into your applications.

### Integration Example (Python)

```python
import requests
import json

# Your deployed model URL
MODEL_URL = "https://sandbox-xyz.e2b.dev"

def predict(input_data):
    """Make a prediction using your deployed model"""
    response = requests.post(
        f"{MODEL_URL}/predict",
        json={"input": input_data},
        headers={"Content-Type": "application/json"}
    )
    return response.json()

def get_model_info():
    """Get information about your model"""
    response = requests.get(f"{MODEL_URL}/info")
    return response.json()

def check_health():
    """Check if model is running"""
    response = requests.get(f"{MODEL_URL}/health")
    return response.json()

# Usage
if __name__ == "__main__":
    # Check health
    health = check_health()
    print(f"Model Status: {health['status']}")
    
    # Get model info
    info = get_model_info()
    print(f"Model Type: {info['modelType']}")
    
    # Make prediction
    result = predict([1, 2, 3, 4])
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
```

### Integration Example (JavaScript)

```javascript
const MODEL_URL = "https://sandbox-xyz.e2b.dev";

async function predict(inputData) {
  const response = await fetch(`${MODEL_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: inputData })
  });
  return response.json();
}

async function getModelInfo() {
  const response = await fetch(`${MODEL_URL}/info`);
  return response.json();
}

async function checkHealth() {
  const response = await fetch(`${MODEL_URL}/health`);
  return response.json();
}

// Usage
(async () => {
  const health = await checkHealth();
  console.log(`Model Status: ${health.status}`);
  
  const info = await getModelInfo();
  console.log(`Model Type: ${info.modelType}`);
  
  const result = await predict([1, 2, 3, 4]);
  console.log(`Prediction: ${result.prediction}`);
  console.log(`Confidence: ${result.confidence}`);
})();
```

### Integration Example (cURL)

```bash
# Check health
curl -X GET https://sandbox-xyz.e2b.dev/health

# Get model info
curl -X GET https://sandbox-xyz.e2b.dev/info

# Make prediction
curl -X POST https://sandbox-xyz.e2b.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3, 4]}'
```

---

## Examples

### Complete Workflow Example

```python
import requests
import time

API_BASE = "https://api.zehanxtech.com/api"
TOKEN = "your_api_token"
USER_ID = "your_user_id"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Step 1: Generate model
print("Generating model...")
response = requests.post(
    f"{API_BASE}/ai/generate",
    json={
        "prompt": "Create a sentiment analysis model",
        "userId": USER_ID
    },
    headers=headers
)
result = response.json()
deployment_url = result['deploymentUrl']
model_id = result['modelId']
print(f"Model deployed at: {deployment_url}")

# Step 2: Wait for model to be ready
print("Waiting for model to be ready...")
time.sleep(5)

# Step 3: Check health
print("Checking model health...")
health_response = requests.get(f"{deployment_url}/health")
print(f"Health: {health_response.json()}")

# Step 4: Make predictions
print("Making predictions...")
predict_response = requests.post(
    f"{deployment_url}/predict",
    json={"input": "This movie is amazing!"},
    headers={"Content-Type": "application/json"}
)
prediction = predict_response.json()
print(f"Prediction: {prediction}")

# Step 5: Check usage
print("Checking usage...")
usage_response = requests.get(
    f"{API_BASE}/usage?userId={USER_ID}",
    headers=headers
)
usage = usage_response.json()
print(f"Tokens used this month: {usage['tokensUsed']}")
print(f"Cost this month: ${usage['costThisMonth']}")
```

---

## Rate Limiting

- **Free Plan**: 1,000 requests/month
- **Pro Plan**: 100,000 requests/month
- **Enterprise**: Unlimited

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1637100000
```

---

## Support

For API support:
- Email: support@zehanxtech.com
- Documentation: https://docs.zehanxtech.com
- Status Page: https://status.zehanxtech.com

---

## Version History

### v1.0.0 (Current)
- Initial API release
- Model generation and deployment
- Usage tracking
- E2B sandbox integration on port 49999

---

## Changelog

### Latest Updates
- Added custom dataset upload support
- Added custom model upload support
- Implemented usage tracking dashboard
- Enhanced billing system with three tiers
- Improved error messages and debugging

---

**Last Updated**: November 16, 2025
**API Version**: 1.0.0
