# 🚀 Quick Setup Guide

## Step 1: Set Up Environment Variables

**Option A: Copy and edit**
```bash
# Copy the example file
cp .env.example .env.local
```

**Option B: Create manually**
Create a new file called `.env.local` in the root directory.

Then edit `.env.local` and paste your API keys:

```bash
# Required
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...

# At least ONE of these AI providers
GROQ_API_KEY=gsk_...           # Fast & Free (recommended to start)
GEMINI_API_KEY=AIza...         # Google's latest
DEEPSEEK_API_KEY=sk-...        # Code-specialized

# Required for sandbox
E2B_API_KEY=e2b_...
```

## Step 2: Install Dependencies

```bash
npm install
```

## Step 3: Run Development Server

```bash
npm run dev
```

## Step 4: Test It Out!

1. Go to http://localhost:3000/login
2. Login (redirects to `/ai-workspace`)
3. Select an AI model from the dropdown (top left)
4. Try this prompt:

```
Create a sentiment analysis model using BERT for analyzing product reviews. 
Use the IMDB dataset and deploy as a REST API with a /predict endpoint.
```

5. Watch the magic happen! ✨

## 🎯 What You'll See

1. **AI generates code** (streaming in real-time)
2. **Files created**: requirements.txt, train.py, app.py
3. **E2B sandbox** created
4. **Dependencies installed** (pip install)
5. **Model trains** (with live logs)
6. **API deployed** (get live URL)
7. **Preview on right side** (iframe showing your API)

## 🔑 Getting API Keys

### Groq (Recommended - Fast & Free)
1. Go to https://console.groq.com/keys
2. Sign up with Google/GitHub
3. Click "Create API Key"
4. Copy and paste into `.env.local`

### Gemini
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Get API key"
4. Copy and paste into `.env.local`

### DeepSeek
1. Go to https://platform.deepseek.com/api_keys
2. Sign up
3. Create API key
4. Copy and paste into `.env.local`

### E2B (Required)
1. Go to https://e2b.dev/dashboard
2. Sign up with GitHub
3. Get your API key from dashboard
4. Copy and paste into `.env.local`

## 🎨 Features

- ✅ **Model Selector**: Switch between Groq, Gemini, DeepSeek
- ✅ **Split View**: Chat on left, sandbox preview on right
- ✅ **Streaming**: Real-time code generation with typing effect
- ✅ **Progress**: Visual indicators for each step (1/7, 2/7, etc.)
- ✅ **Live Logs**: See training output in real-time
- ✅ **Follow-ups**: Continue conversation and iterate
- ✅ **Animations**: Smooth, professional UI

## 🐛 Common Issues

**"Invalid API key"**
- Double-check your API keys in `.env.local`
- Make sure there are no extra spaces
- Restart dev server after changing `.env.local`

**"E2B sandbox creation failed"**
- Check E2B_API_KEY
- Verify you have credits at https://e2b.dev/dashboard
- E2B free tier: 100 hours/month

**"Training failed"**
- This is normal for complex models (GPU required)
- Try simpler models first
- Check the terminal output in chat for errors

**"No preview showing"**
- Wait for all 7 steps to complete
- Check if deployment step succeeded
- Click "Open in New Tab" button

## 📝 Example Prompts

**Beginner:**
```
Create a simple text classifier using scikit-learn. 
Train on sample data and deploy with FastAPI.
```

**Intermediate:**
```
Build a sentiment analysis model using BERT. 
Fine-tune on IMDB reviews and add confidence scores to API.
```

**Advanced:**
```
Create a multi-label image classifier using ResNet50. 
Use transfer learning, data augmentation, and deploy with 
batch prediction support via FastAPI.
```

## 🎉 You're All Set!

Start building amazing AI models with just a prompt! 🚀

Need help? Check `AI_WORKSPACE_README.md` for detailed documentation.
