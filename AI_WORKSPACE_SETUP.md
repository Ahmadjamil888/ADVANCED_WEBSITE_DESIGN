# zehanx AI Workspace Setup Guide

## Overview
The AI Workspace is a comprehensive platform for generating, training, and deploying custom AI models. It features a ChatGPT-like interface with specialized tools for AI development.

## 🗄️ Database Setup

### 1. Run the SQL Schema
Execute the SQL schema in your Supabase SQL Editor:
```bash
# Navigate to your Supabase project dashboard
# Go to SQL Editor
# Copy and paste the contents of: database/ai_workspace_schema.sql
# Click "Run" to create all tables and policies
```

### 2. Enable Required Extensions
The schema automatically enables:
- `uuid-ossp` for UUID generation
- `vector` for embeddings (if available)

## 🔧 Environment Configuration

### Required Environment Variables
Add these to your `.env.local`:

```env
# Supabase (already configured)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# AI Services
GEMINI_API_KEY=your_gemini_api_key

# Inngest Configuration
INNGEST_SIGNING_KEY=signkey-prod-4b628d68eb7ff4117cf134da546244096dad4450adfd9518fb6b4cb569ee48c1
INNGEST_EVENT_KEY=your-inngest-event-key

# E2B Sandbox (for code execution)
E2B_API_KEY=your-e2b-api-key

# External APIs
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
```

## 📦 Installation

### 1. Install Dependencies
```bash
cd DHAMIA
npm install inngest
```

### 2. Configure Inngest
Your Inngest endpoint will be available at:
```
https://zehanxtech.com/api/inngest
```

Register this URL in your Inngest dashboard with the signing key provided.

## 🚀 Features

### AI Workspace Interface
- **ChatGPT-like Interface**: Conversational AI interaction
- **Multiple Modes**: Chat, Code, Models, Fine-tune, Research, App Builder, Translate
- **Context Panel**: File uploads, entity memory, tools management
- **Sidebar**: Chat history, pinned conversations, quick actions

### AI Model Generation
1. **Describe Your Model**: Tell the AI what kind of model you want
2. **Automatic Code Generation**: Complete PyTorch/TensorFlow implementations
3. **Dataset Integration**: Automatic dataset finding from Kaggle/Hugging Face
4. **Training Scripts**: Ready-to-run training code with proper configurations
5. **Deployment**: One-click deployment to Hugging Face

### Supported Model Types
- **Text Classification**: Sentiment analysis, document classification
- **Image Classification**: Object detection, image recognition
- **Natural Language Processing**: Chatbots, text generation
- **Computer Vision**: CNN architectures, transfer learning
- **Time Series**: LSTM, forecasting models
- **Custom Architectures**: Flexible model creation

## 🔄 Workflow

### 1. Create AI Model
```
User: "Create a sentiment analysis model using BERT for movie reviews"
```

The AI will:
1. Generate complete PyTorch code
2. Find suitable dataset (IMDB reviews)
3. Create training script with hyperparameters
4. Provide requirements.txt
5. Generate README with instructions

### 2. Train Model (via Inngest)
```javascript
// Triggered automatically or manually
await inngest.send({
  name: "ai/model.train",
  data: {
    modelId: "uuid",
    userId: "uuid",
    trainingConfig: { epochs: 10, batch_size: 32 }
  }
});
```

### 3. Deploy to Hugging Face
```javascript
// User provides HF token securely
await inngest.send({
  name: "ai/model.deploy",
  data: {
    modelId: "uuid",
    userId: "uuid",
    hfToken: "hf_token",
    repoName: "username/model-name"
  }
});
```

## 🛠️ Tools & Plugins

### Available Tools
- **SQL Runner**: Execute database queries
- **Code Runner**: Run code in E2B sandbox
- **API Playground**: Test API endpoints
- **Quiz Generator**: Create educational content
- **Model Trainer**: Train AI models
- **Dataset Finder**: Search Kaggle/HuggingFace

### Context Management
- **File Uploads**: PDF, TXT, CSV, JSON support
- **Vector Database**: Automatic indexing and search
- **Entity Memory**: Track important concepts and entities
- **Conversation History**: Persistent chat storage

## 🔐 Security Features

### Row Level Security (RLS)
All tables have RLS policies ensuring users can only access their own data.

### API Key Management
- Encrypted storage of external API keys
- Secure token handling for Hugging Face deployments
- Environment variable protection

## 📊 Database Schema

### Core Tables
- `chats`: Conversation management
- `messages`: Chat history
- `ai_models`: Custom model definitions
- `training_jobs`: Model training status
- `chat_files`: File uploads and context
- `prompt_templates`: Reusable prompts
- `generated_apps`: Built applications
- `user_tools`: Tool configurations

### Relationships
```
users (auth.users)
├── chats
│   ├── messages
│   ├── chat_files
│   └── chat_entities
├── ai_models
│   └── training_jobs
├── prompt_templates
├── generated_apps
├── user_tools
└── user_integrations
```

## 🌐 API Endpoints

### AI Workspace
- `POST /api/ai-workspace/generate` - Generate AI responses
- `GET/POST/PUT /api/inngest` - Inngest webhook handler

### Inngest Functions
- `ai/model.generate` - Generate model code
- `ai/model.train` - Train models in sandbox
- `ai/model.deploy` - Deploy to Hugging Face
- `ai/dataset.find` - Find suitable datasets

## 🎯 Usage Examples

### Example 1: Text Classification
```
User: "Create a text classification model for customer support tickets that can categorize them as urgent, normal, or low priority"

AI Response:
1. Generates BERT-based classifier
2. Finds customer support dataset
3. Creates training script with proper preprocessing
4. Provides evaluation metrics
5. Includes deployment instructions
```

### Example 2: Image Recognition
```
User: "Build an image classification model that can identify different dog breeds"

AI Response:
1. Creates CNN architecture with transfer learning
2. Finds dog breed dataset from Kaggle
3. Generates data augmentation pipeline
4. Provides training monitoring code
5. Includes model optimization tips
```

## 🔧 Customization

### Adding New Model Types
1. Update `generatePyTorchCode()` in `lib/inngest/functions.ts`
2. Add new templates in `components/Composer.tsx`
3. Update mode selector in `components/ModeSelector.tsx`

### Adding New Tools
1. Create tool component
2. Add to `user_tools` table
3. Update context panel configuration

## 📈 Monitoring & Analytics

### Training Job Monitoring
- Real-time progress tracking
- Loss and accuracy metrics
- Error logging and debugging
- Resource usage monitoring

### Usage Analytics
- Model creation statistics
- Training success rates
- User engagement metrics
- Popular model types

## 🚨 Troubleshooting

### Common Issues
1. **Inngest Connection**: Verify signing key and endpoint URL
2. **Database Permissions**: Check RLS policies
3. **API Limits**: Monitor Gemini API usage
4. **Sandbox Errors**: Verify E2B configuration

### Debug Mode
Enable detailed logging by setting:
```env
NODE_ENV=development
DEBUG=inngest:*
```

## 🎉 Getting Started

1. **Run Database Schema**: Execute `ai_workspace_schema.sql`
2. **Install Dependencies**: `npm install inngest`
3. **Configure Environment**: Update `.env.local`
4. **Register Inngest**: Add `https://zehanxtech.com/api/inngest`
5. **Start Development**: `npm run dev`
6. **Test AI Generation**: Create a new chat and describe your model

## 🤝 Support

For issues or questions:
- Check the troubleshooting section
- Review Inngest logs
- Verify database connections
- Test API endpoints individually

---

**Built with ❤️ by zehanxtech**
*AI for the betterment of humanity*