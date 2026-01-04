# Quick Start Guide - AI Model Generator

## What's New

### 1. Port Configuration (49999)
Your AI models now deploy on **port 49999** - the standard E2B sandbox model backend port.

**Why?** This is the dedicated port for model serving in E2B sandboxes, ensuring reliable and consistent model deployment.

### 2. Modern Dashboard Design
The dashboard now features:
- **Deep black theme** (#0a0a0a) for reduced eye strain
- **Purple gradient accents** on interactive elements
- **Smooth hover effects** with lift animations
- **Closable sidebar** with X button for better space management

### 3. Enhanced Billing System
- Three-tier pricing: Free, Pro, Enterprise
- Real-time model usage tracking
- Stripe integration for seamless upgrades
- API access indicator

---

## Getting Started

### Prerequisites
```env
E2B_API_KEY=your_e2b_api_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
STRIPE_SECRET_KEY=your_stripe_key
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your_stripe_public_key
```

### Running the Application
```bash
npm install
npm run dev
```

Visit `http://localhost:3000` and navigate to `/ai-workspace` for the dashboard.

---

## Dashboard Navigation

### Sidebar
- **🤖 Trained Models**: View all your generated AI models
- **🧠 LLMs**: Access large language models
- **📊 Datasets**: Manage your datasets
- **⚙️ In Progress**: Monitor active training jobs
- **💳 Billing**: Manage your subscription plan

**Tip**: Hover over the sidebar to expand it, or click the X button to collapse.

---

## Creating Your First AI Model

### Step 1: Navigate to Trained Models
Click the "Trained Models" section in the sidebar.

### Step 2: Click "Create AI Model"
The purple gradient button in the top-right corner.

### Step 3: Enter Your Prompt
Describe what AI model you want to create. Examples:
- "Create a sentiment analysis model"
- "Build a text classification model for product reviews"
- "Generate a neural network for image classification"

### Step 4: Wait for Deployment
The system will:
1. Generate code using AI
2. Create an E2B sandbox
3. Train your model
4. Deploy it on port 49999

### Step 5: Access Your Model
Once deployed, you'll get a URL like:
```
https://sandbox-xyz.e2b.dev/
```

Your model endpoints:
- `/health` - Check if model is running
- `/predict` - Make predictions
- `/info` - Get model information

---

## Making Predictions

### Using cURL
```bash
curl -X POST https://sandbox-xyz.e2b.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3, 4]}'
```

### Using Python
```python
import requests

url = "https://sandbox-xyz.e2b.dev/predict"
data = {"input": [1, 2, 3, 4]}
response = requests.post(url, json=data)
print(response.json())
```

### Using JavaScript
```javascript
const response = await fetch('https://sandbox-xyz.e2b.dev/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input: [1, 2, 3, 4] })
});
const result = await response.json();
console.log(result);
```

---

## Billing Plans

### Free Plan - $0/month
- 1 AI Model
- Basic Support
- Community Access

### Pro Plan - $50/month
- 10 AI Models
- Priority Support
- Email Support

### Enterprise Plan - $450/month
- 30 AI Models
- API Access
- 24/7 Support
- Custom Integration

---

## Troubleshooting

### Model Not Responding
1. Check the deployment URL is correct
2. Wait 30 seconds after deployment
3. Try the `/health` endpoint first
4. Check browser console for errors

### Port 49999 Not Accessible
- Ensure E2B_API_KEY is set correctly
- Check firewall settings
- Verify sandbox is still running
- Check E2B dashboard for sandbox status

### Training Failed
- Review the error message in the dashboard
- Check if your prompt is specific enough
- Try a simpler model first
- Verify all dependencies are available

### Billing Issues
- Check your Stripe account
- Verify payment method is valid
- Contact support for assistance

---

## Key Features

✅ **AI-Powered Code Generation** - Uses Groq, Gemini, or DeepSeek
✅ **Automatic Deployment** - One-click model deployment
✅ **Real-time Monitoring** - Watch training progress live
✅ **REST API** - Access models via HTTP endpoints
✅ **Secure Sandboxes** - Isolated E2B environments
✅ **Model Versioning** - Track model history
✅ **Usage Analytics** - Monitor model performance
✅ **Team Collaboration** - Share models with team members

---

## API Reference

### Health Check
```
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

### Make Prediction
```
POST /predict
Body: {"input": [...]}
Response: {"success": true, "prediction": [...], "confidence": 0.95}
```

### Model Info
```
GET /info
Response: {
  "model_type": "custom",
  "model_loaded": true,
  "endpoints": ["/health", "/predict", "/info"]
}
```

---

## Documentation

For detailed information, see:
- **Port Configuration**: `MODEL_BACKEND_PORT_GUIDE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **API Documentation**: Check `/api` routes in source code

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation files
3. Check browser console for error messages
4. Contact support team

---

## Next Steps

1. ✅ Set up environment variables
2. ✅ Run the application
3. ✅ Create your first model
4. ✅ Test the prediction endpoints
5. ✅ Upgrade to Pro for more models
6. ✅ Integrate into your application

Happy model building! 🚀
