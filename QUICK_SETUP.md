# 🚀 Quick Setup Guide

## Current Status

✅ **Fixed**:
- E2B template error (removed 'python3' parameter)
- Code viewer with tabs
- Toggle between Code and Sandbox
- Theme toggle (light/dark)
- Sign out button
- Deep dark theme
- CSS modules (no Tailwind)

⏳ **In Progress**:
- Prisma installation
- Database schema setup
- Authentication implementation
- Message saving

❌ **Still Need to Fix**:
1. Code appearing in chat (should only be in Code tab)
2. Real authentication (currently just redirects)
3. Message/chat saving to database
4. Project management

---

## Immediate Next Steps

### 1. Complete Prisma Installation

```bash
# Wait for current installation to finish, then:
npx prisma generate
npx prisma db push
```

### 2. Update `.env.local`

Add your DATABASE_URL:
```env
DATABASE_URL=postgresql://postgres.xxx:xxx@aws-0-us-east-1.pooler.supabase.com:6543/postgres?pgbouncer=true
```

Get this from Supabase Dashboard → Project Settings → Database → Connection String (Pooler)

### 3. Apply Database Schema

**Option A: Using Prisma (Recommended)**
```bash
npx prisma db push
```

**Option B: Manual SQL**
1. Go to Supabase Dashboard → SQL Editor
2. Copy contents of `database/schema.sql`
3. Run the SQL

### 4. Critical Code Fixes Needed

#### Fix 1: Remove Code from Chat Display

**File**: `src/app/ai-workspace/page.tsx`

Find this code around line 108-116:
```typescript
case 'ai-stream':
  fullResponse += data.data.content;
  setStreamingContent(fullResponse);  // ❌ REMOVE THIS LINE
  // Parse files from streaming content in real-time
  const parsedFiles = parseFilesFromContent(fullResponse);
  if (Object.keys(parsedFiles).length > 0) {
    setGeneratedFiles(parsedFiles);
  }
  break;
```

Change to:
```typescript
case 'ai-stream':
  fullResponse += data.data.content;
  // DON'T show code in chat, only in Code tab
  const parsedFiles = parseFilesFromContent(fullResponse);
  if (Object.keys(parsedFiles).length > 0) {
    setGeneratedFiles(parsedFiles);
  }
  break;
```

#### Fix 2: Remove Streaming Code Display

**File**: `src/app/ai-workspace/page.tsx`

Find this code around line 233-239:
```typescript
{streamingContent && (
  <ChatMessage
    role="assistant"
    content={streamingContent}
    isStreaming={true}
  />
)}
```

**REMOVE** this entire block. Code should only appear in the Code tab, not in chat.

#### Fix 3: Update Message Display

Messages should only show:
- ✅ User prompts
- ✅ Status messages ("Generating code...", "Training model...")
- ✅ Completion messages ("Model trained successfully!")
- ❌ NO code blocks
- ❌ NO file contents

---

## Authentication Setup

### 1. Update Supabase Client

**File**: `src/lib/supabase.ts`

```typescript
import { createClient } from '@supabase/supabase-js';

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);
```

### 2. Enable Auth in Supabase

1. Go to Supabase Dashboard → Authentication → Providers
2. Enable Email provider
3. Enable Google OAuth (optional)
4. Set redirect URL: `http://localhost:3000/ai-workspace`

### 3. Update Login Page

The login page needs to actually call Supabase auth instead of just redirecting.

---

## Database Schema Overview

### Tables

1. **Project** - User workspaces/chats
   - `id`: UUID
   - `name`: Project name
   - `userId`: Owner
   - `messages`: Related messages

2. **Message** - Chat messages
   - `id`: UUID
   - `content`: Message text
   - `role`: USER | ASSISTANT
   - `type`: RESULT | ERROR
   - `projectId`: Parent project
   - `Fragment`: Optional sandbox result

3. **Fragment** - Sandbox results
   - `id`: UUID
   - `messageId`: Parent message
   - `sandboxUrl`: E2B sandbox URL
   - `title`: Fragment title
   - `files`: JSON of generated files

4. **Usage** - API usage tracking
   - `key`: Usage key
   - `points`: Usage points
   - `expire`: Expiration date

---

## Testing After Setup

### 1. Test Authentication
```bash
npm run dev
```
- Go to `/login`
- Try to login (should work with real auth)
- Should redirect to `/ai-workspace`
- Try to access `/ai-workspace` without login (should redirect to `/login`)

### 2. Test Code Generation
- Enter prompt: "Create a sentiment analysis model"
- Code should appear in **Code tab** (right side)
- Chat should show **status messages only**
- No code blocks in chat

### 3. Test Database Saving
- Check Supabase Dashboard → Table Editor
- Should see new Project created
- Should see Messages saved
- Should see Fragment with sandbox URL

### 4. Test E2B Sandbox
- Code should be written to sandbox
- Training should run
- API should deploy
- Sandbox URL should be saved
- Toggle to "Sandbox" tab to see live preview

---

## Common Issues

### Issue: Prisma Client Not Found
**Solution**:
```bash
npx prisma generate
```

### Issue: Database Connection Error
**Solution**: Check your DATABASE_URL in `.env.local`

### Issue: Code Still Showing in Chat
**Solution**: Remove `setStreamingContent()` call and the streaming message display component

### Issue: E2B 403 Error
**Solution**: Already fixed - we removed the 'python3' template parameter

### Issue: Authentication Not Working
**Solution**: Make sure Supabase URL and keys are correct in `.env.local`

---

## File Checklist

- [x] `prisma/schema.prisma` - Created
- [x] `database/schema.sql` - Created
- [ ] Run `npx prisma generate`
- [ ] Run `npx prisma db push`
- [ ] Update `src/app/ai-workspace/page.tsx` - Remove code from chat
- [ ] Update `src/app/login/page.tsx` - Add real auth
- [ ] Update `src/app/api/ai/generate/route.ts` - Save to database
- [ ] Test everything

---

## Priority Fixes (Do These First!)

1. **Remove `setStreamingContent(fullResponse)`** from ai-stream case
2. **Remove streaming code display** from chat UI
3. **Run `npx prisma generate`**
4. **Run `npx prisma db push`**
5. **Test that code only appears in Code tab**

---

## Expected Behavior After Fixes

### Chat Area (Left Side)
```
User: Create a sentiment analysis model

🤖 Analyzing your request with Llama 3.3 70B...
📝 Extracting generated files...
⚡ Creating E2B sandbox environment...
📂 Writing files to sandbox...
📦 Installing dependencies...
🏋️ Training model... This may take a few minutes.
🚀 Deploying FastAPI server...
✅ All done! Your model is trained and deployed.
```

### Code Tab (Right Side)
```
[requirements.txt] [config.json] [train.py] [app.py]

torch==2.1.0
transformers==4.35.0
datasets==2.14.0
...
```

### Sandbox Tab (Right Side)
```
[Live E2B Sandbox Preview]
iframe showing deployed API
```

---

## Need Help?

1. Check `IMPLEMENTATION_PLAN.md` for detailed steps
2. Check `FINAL_FIXES.md` for what was already fixed
3. Check `CHANGES_SUMMARY.md` for all changes made
4. Check console for errors
5. Check Supabase logs for database errors

---

## Ready to Deploy?

After all fixes:
```bash
# Build
npm run build

# Deploy to Vercel
vercel --prod
```

Make sure to add all environment variables in Vercel dashboard!
