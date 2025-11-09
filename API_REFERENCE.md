# 🔌 API Reference

## AI Provider Endpoints

### Groq (OpenAI-compatible)
```
Base URL: https://api.groq.com/openai/v1
Endpoint: /chat/completions
Method: POST
Auth: Bearer token
```

**Example:**
```typescript
const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${GROQ_API_KEY}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello!' }
    ],
    stream: true,
  }),
});
```

### Gemini
```
Base URL: https://generativelanguage.googleapis.com/v1beta
Endpoint: /models/{model}:streamGenerateContent?key={API_KEY}
Method: POST
Auth: API key in URL
```

**Example:**
```typescript
const response = await fetch(
  `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?key=${GEMINI_API_KEY}`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Hello!' }]
        }
      ],
    }),
  }
);
```

### DeepSeek (OpenAI-compatible)
```
Base URL: https://api.deepseek.com
Endpoint: /chat/completions
Method: POST
Auth: Bearer token
```

**Example:**
```typescript
const response = await fetch('https://api.deepseek.com/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'deepseek-chat',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello!' }
    ],
    stream: true,
  }),
});
```

## E2B Sandbox API

### Create Sandbox
```typescript
import { Sandbox } from '@e2b/code-interpreter';

const sandbox = await Sandbox.create('python3');
await sandbox.setTimeout(1800000); // 30 minutes
const sandboxId = sandbox.sandboxId;
```

### Write Files
```typescript
await sandbox.files.write('/home/user/train.py', pythonCode);
```

### Run Commands
```typescript
const result = await sandbox.commands.run('python train.py', {
  onStdout: (data) => console.log(data),
  onStderr: (data) => console.error(data),
});
```

### Get Sandbox URL
```typescript
const host = sandbox.getHost(8000); // port number
const url = `http://${host}`;
```

## Our API Endpoint

### POST /api/ai/generate

Generates AI model code, trains it, and deploys to E2B.

**Request:**
```json
{
  "prompt": "Create a sentiment analysis model using BERT",
  "modelKey": "llama-3.3-70b",
  "chatId": "optional-chat-id",
  "userId": "optional-user-id"
}
```

**Response (Server-Sent Events):**
```
data: {"type":"status","data":{"message":"🤖 Initializing...","step":1,"total":7}}

data: {"type":"ai-stream","data":{"content":"I'll create..."}}

data: {"type":"files","data":{"files":["requirements.txt","train.py","app.py"]}}

data: {"type":"sandbox","data":{"sandboxId":"abc123"}}

data: {"type":"training","data":{"output":"Epoch 1/3..."}}

data: {"type":"deployment-url","data":{"url":"http://abc123.e2b.dev"}}

data: {"type":"complete","data":{"message":"✅ Done!","deploymentUrl":"..."}}
```

## Environment Variables

```bash
# AI Providers
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...

# E2B
E2B_API_KEY=e2b_...

# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://...
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```

## Model IDs

### Groq
- `llama-3.3-70b-versatile` - Fast, powerful
- `llama-3.1-8b-instant` - Ultra fast

### Gemini
- `gemini-2.0-flash-exp` - Latest, multimodal
- `gemini-1.5-pro` - Advanced reasoning

### DeepSeek
- `deepseek-chat` - General purpose
- `deepseek-coder` - Code-specialized

## Rate Limits

- **Groq**: 30 requests/minute (free tier)
- **Gemini**: 60 requests/minute (free tier)
- **DeepSeek**: 60 requests/minute (free tier)
- **E2B**: 100 hours/month (free tier)

## Error Handling

All API calls should handle these errors:

```typescript
try {
  const response = await fetch(...);
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
} catch (error) {
  console.error('API call failed:', error);
  // Handle error appropriately
}
```
