# 🚀 AI Model Training Studio

A complete rebuild of the AI workspace using direct AI API calls (Groq, Gemini, DeepSeek) and E2B sandboxes.

## ✨ Features

- **Multiple AI Models**: Choose from Groq (Llama), Gemini, or DeepSeek models
- **Real-time Streaming**: Watch AI generate code in real-time with smooth animations
- **Live Sandbox Preview**: See your trained model running in E2B sandbox (split-view like vibe project)
- **Follow-up Prompts**: Continue conversations and iterate on your models
- **Progress Tracking**: Visual progress indicators for each step (7 steps total)
- **File Generation**: Automatically creates requirements.txt, train.py, app.py, config.json
- **Live Deployment**: Get a working API URL after training completes

## 🎯 How It Works

1. **User selects AI model** (Groq/Gemini/DeepSeek) from dropdown
2. **User enters prompt** describing the model they want
3. **AI generates code** - streaming response with file generation
4. **E2B sandbox created** - isolated Python environment
5. **Files written** - all generated code saved to sandbox
6. **Dependencies installed** - pip install requirements.txt
7. **Model trains** - python train.py with real-time logs
8. **API deployed** - FastAPI server starts, returns live URL

## 📁 New File Structure

```
src/
├── lib/ai/
│   ├── models.ts          # AI model configurations
│   ├── client.ts          # Streaming AI client (Groq/Gemini/DeepSeek)
│   └── prompts.ts         # System prompts for code generation
├── app/
│   ├── api/ai/generate/
│   │   └── route.ts       # Main API endpoint (replaces Inngest)
│   └── ai-workspace/
│       ├── new-page.tsx   # New UI (use this!)
│       ├── components/
│       │   ├── ModelSelector.tsx
│       │   ├── SandboxPreview.tsx
│       │   ├── ChatMessage.tsx
│       │   └── StatusIndicator.tsx
│       └── animations.css
```

## 🔑 Environment Variables

Copy `.env.example` to `.env.local` and fill in your API keys:

```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_key

# AI Models (get from respective dashboards)
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...

# E2B Sandbox
E2B_API_KEY=e2b_...
```

### Where to Get API Keys:

- **Groq**: https://console.groq.com/keys
- **Gemini**: https://aistudio.google.com/app/apikey
- **DeepSeek**: https://platform.deepseek.com/api_keys
- **E2B**: https://e2b.dev/dashboard

## 🚀 Usage

1. **Install dependencies**:
```bash
npm install
```

2. **Set up environment variables** in `.env.local`

3. **Run development server**:
```bash
npm run dev
```

4. **Navigate to** http://localhost:3000/ai-workspace

5. **Select your AI model** from the dropdown (top left)

6. **Enter a prompt** like:
   - "Create a sentiment analysis model using BERT"
   - "Build an image classifier with ResNet"
   - "Train a text generation model with GPT-2"

7. **Watch the magic happen**:
   - AI generates code (streaming)
   - Files created in E2B sandbox
   - Dependencies installed
   - Model trains with live logs
   - API deployed with live URL

8. **Test your model** using the returned URL!

## 🎨 UI Features

- **Split View**: Chat on left, sandbox preview on right (like vibe project)
- **Smooth Animations**: Typing effect for AI responses
- **Progress Indicators**: Visual progress bars for each step
- **File Badges**: See which files were generated
- **Live Terminal**: Real-time output from training
- **Sandbox Preview**: iframe showing your deployed API

## 🔄 Switching to New Page

To use the new UI, rename the files:

```bash
# Backup old page
mv src/app/ai-workspace/page.tsx src/app/ai-workspace/page.old.tsx

# Use new page
mv src/app/ai-workspace/new-page.tsx src/app/ai-workspace/page.tsx
```

## 🗑️ What Was Removed

- ❌ Inngest (all files deleted)
- ❌ Fixed function definitions
- ❌ Complex workflow orchestration
- ❌ Multiple API routes

## ✅ What Was Added

- ✅ Direct AI API streaming
- ✅ Model selector dropdown
- ✅ Real-time code generation
- ✅ Smooth animations
- ✅ Split-view UI
- ✅ Follow-up prompt support

## 🐛 Troubleshooting

**AI not responding?**
- Check API keys in `.env.local`
- Verify you have credits/quota for the selected model

**E2B sandbox fails?**
- Check E2B_API_KEY
- Verify you have E2B credits

**Training fails?**
- Check terminal output in the chat
- Some models require GPU (not available in E2B free tier)

**No preview showing?**
- Wait for deployment step to complete
- Check if FastAPI server started successfully

## 📝 Example Prompts

```
Create a sentiment analysis model using BERT for analyzing customer reviews. 
Train on IMDB dataset and deploy as a REST API.

Build an image classifier using ResNet50 to classify cats vs dogs. 
Use transfer learning and deploy with FastAPI.

Train a text generation model using GPT-2 on custom dataset. 
Add temperature control to the API endpoint.

Create a question-answering system using T5 model. 
Fine-tune on SQuAD dataset and expose via API.
```

## 🎯 Next Steps

1. Add more AI models (Claude, OpenAI, etc.)
2. Add file upload for custom datasets
3. Add model comparison view
4. Add training metrics visualization
5. Add model versioning
6. Add deployment to production (not just E2B)

---

**Built with**: Next.js, TypeScript, Tailwind CSS, E2B, Groq/Gemini/DeepSeek APIs
