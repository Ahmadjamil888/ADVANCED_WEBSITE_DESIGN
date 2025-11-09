# 🚀 Supabase Setup Guide

## ✅ Changes Made

1. **Removed Prisma** - No more PostgreSQL connection string needed
2. **Using Supabase Client** - Direct Supabase integration
3. **Complete Schema** - All tables from your schema included
4. **Row Level Security** - Proper RLS policies for data protection
5. **Auto User Creation** - Trigger creates user profile on signup

---

## 📋 Step-by-Step Setup

### 1. Apply Database Schema

Go to your Supabase Dashboard:
1. Click on **SQL Editor** in the left sidebar
2. Click **New Query**
3. Copy the entire contents of `database/supabase_schema.sql`
4. Paste into the SQL editor
5. Click **Run** (or press Ctrl+Enter)

This will create:
- ✅ All tables (users, chats, messages, fragments, ai_models, etc.)
- ✅ Indexes for performance
- ✅ Row Level Security policies
- ✅ Triggers for auto-updating timestamps
- ✅ Auto user profile creation on signup

### 2. Enable Authentication

In Supabase Dashboard:
1. Go to **Authentication** → **Providers**
2. Enable **Email** provider
3. (Optional) Enable **Google** OAuth:
   - Add your Google Client ID and Secret
   - Set redirect URL: `http://localhost:3000/ai-workspace`

### 3. Update Environment Variables

Update your `.env.local`:

```env
# Supabase (get from Dashboard → Settings → API)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here

# AI APIs
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

# E2B
E2B_API_KEY=your_e2b_key
```

### 4. Test Database Connection

Run this test:
```bash
npm run dev
```

Then in browser console:
```javascript
// Test Supabase connection
const { data, error } = await supabase.from('users').select('count');
console.log(data, error);
```

---

## 📊 Database Structure

### Core Tables

#### `users`
- Extends `auth.users`
- Stores user profile data
- Tracks usage stats

#### `chats`
- User workspaces/projects
- Stores chat settings
- Links to messages

#### `messages`
- Chat messages (USER/ASSISTANT)
- Stores AI responses
- Links to fragments

#### `fragments`
- Sandbox results
- Generated code files
- E2B sandbox URLs

#### `ai_models`
- User's trained models
- Training configuration
- Performance metrics

#### `training_jobs`
- Training job status
- Progress tracking
- Logs and errors

---

## 🔐 Row Level Security (RLS)

All tables have RLS enabled with policies:

**Users can only**:
- View their own data
- Create data for themselves
- Update their own data
- Delete their own data

**Example**: A user can only see chats they created, messages in their chats, and fragments from their messages.

---

## 🛠️ Using the Database

### Import the client:
```typescript
import { db, supabase } from '@/lib/db';
```

### Create a chat:
```typescript
const chat = await db.createChat(userId, 'My AI Project');
```

### Create a message:
```typescript
const message = await db.createMessage(
  chatId,
  'USER',
  'Create a sentiment analysis model'
);
```

### Create a fragment (sandbox result):
```typescript
const fragment = await db.createFragment(
  messageId,
  sandboxUrl,
  sandboxId,
  'Sentiment Model',
  { 'train.py': '...', 'app.py': '...' }
);
```

### Get chat history:
```typescript
const messages = await db.getMessages(chatId);
// Returns messages with fragments included
```

---

## 🔄 Auto User Creation

When a user signs up, a trigger automatically creates their profile in the `users` table:

```sql
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

This means:
- ✅ No manual user creation needed
- ✅ Username auto-generated from email
- ✅ Profile ready immediately after signup

---

## 📝 TypeScript Types

All database types are defined in `src/lib/db.ts`:

```typescript
import type { Chat, Message, Fragment, User } from '@/lib/db';

const chat: Chat = {
  id: '...',
  user_id: '...',
  title: 'My Chat',
  // ... other fields
};
```

---

## 🧪 Testing Queries

### Test in Supabase SQL Editor:

```sql
-- Check if tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Check RLS policies
SELECT * FROM pg_policies WHERE schemaname = 'public';

-- Test user creation (after signup)
SELECT * FROM public.users;

-- Test chat creation
INSERT INTO public.chats (user_id, title)
VALUES ('your-user-id', 'Test Chat')
RETURNING *;
```

---

## ⚠️ Common Issues

### Issue: "relation does not exist"
**Solution**: Run the schema SQL in Supabase SQL Editor

### Issue: "new row violates row-level security policy"
**Solution**: Make sure you're authenticated and RLS policies are applied

### Issue: "permission denied for table"
**Solution**: Check that RLS is enabled and policies exist

### Issue: "User profile not created"
**Solution**: Check that the trigger `on_auth_user_created` exists

---

## 🎯 Next Steps

After database setup:

1. ✅ Schema applied
2. ✅ Auth enabled
3. ✅ Environment variables set
4. ⏳ Update API route to save messages
5. ⏳ Update UI to load chat history
6. ⏳ Implement authentication
7. ⏳ Test end-to-end flow

---

## 📚 Additional Tables

Your schema includes many additional tables for advanced features:

- `api_keys` - API key management
- `billing` - Subscription/credits
- `chat_entities` - NER/entity extraction
- `chat_files` - File uploads
- `generated_apps` - Generated applications
- `model_usage` - Usage tracking
- `prompt_templates` - Saved prompts
- `rate_limits` - API rate limiting
- `usage_logs` - Detailed logging
- `user_integrations` - Third-party integrations
- `user_sessions` - Session management
- `user_tools` - User tool preferences

All tables are ready to use!

---

## 🔗 Useful Links

- [Supabase Dashboard](https://supabase.com/dashboard)
- [Supabase Docs](https://supabase.com/docs)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)
- [Supabase Auth](https://supabase.com/docs/guides/auth)

---

## ✨ Schema Features

- ✅ UUID primary keys
- ✅ Timestamps (created_at, updated_at)
- ✅ JSONB for flexible data
- ✅ Foreign key constraints
- ✅ Cascade deletes
- ✅ Indexes for performance
- ✅ Row Level Security
- ✅ Auto-updating triggers
- ✅ User profile auto-creation

**Everything is ready to use!** 🎉
