# 🚀 Complete Implementation Plan

## Issues to Fix

### 1. ❌ Code Appearing in Chat
**Problem**: AI-generated code is showing in the chat/prompt area
**Solution**: 
- Code should ONLY appear in the "Code" tab on the right
- Chat should only show status messages and user prompts
- No code blocks in chat messages

### 2. ❌ E2B Sandbox Error
**Problem**: Still getting E2B template errors
**Solution**: Already fixed by removing `'python3'` parameter

### 3. ❌ No Real Authentication
**Problem**: Login just redirects without actual auth
**Solution**: Implement proper Supabase authentication with:
- Email/password login
- Google OAuth
- Session management
- Protected routes

### 4. ❌ No Chat/Message Saving
**Problem**: Messages aren't saved to database
**Solution**: Implement Prisma + Supabase integration:
- Save projects (workspaces)
- Save messages
- Save fragments (sandbox results)
- Link everything properly

### 5. ❌ Not Following Vibe Project Structure
**Problem**: Current implementation doesn't match zehanxtech/vibe
**Solution**: Copy exact structure:
- Use Prisma ORM
- Use same database schema
- Use same E2B integration pattern
- Use same message/fragment structure

---

## Database Schema

### Tables (from Vibe project)
1. **Project** - User workspaces/chats
2. **Message** - Chat messages (USER/ASSISTANT)
3. **Fragment** - Sandbox results with files
4. **Usage** - API usage tracking

### Enums
- `MessageRole`: USER, ASSISTANT
- `MessageType`: RESULT, ERROR

---

## Implementation Steps

### Phase 1: Database Setup ✅
- [x] Create `prisma/schema.prisma`
- [x] Create `database/schema.sql`
- [ ] Install Prisma (`npm install @prisma/client prisma`)
- [ ] Run `npx prisma generate`
- [ ] Run `npx prisma db push` (or apply SQL manually to Supabase)

### Phase 2: Authentication 🔄
- [ ] Update `.env.local` with Supabase credentials
- [ ] Create proper Supabase client
- [ ] Implement auth middleware
- [ ] Update login page with real auth
- [ ] Add session management
- [ ] Protect `/ai-workspace` route

### Phase 3: Message Saving 🔄
- [ ] Create Prisma client singleton
- [ ] Update API route to save messages
- [ ] Save user messages to DB
- [ ] Save AI responses to DB
- [ ] Link messages to projects
- [ ] Save fragments with sandbox results

### Phase 4: UI Fixes 🔄
- [ ] Remove code from chat display
- [ ] Only show status messages in chat
- [ ] Keep code ONLY in Code tab
- [ ] Add project/workspace selector
- [ ] Add chat history sidebar
- [ ] Add "New Project" button

### Phase 5: E2B Integration 🔄
- [ ] Copy E2B pattern from vibe project
- [ ] Use `Sandbox.create()` without template
- [ ] Save sandbox URLs to fragments
- [ ] Link fragments to messages
- [ ] Display fragments in UI

---

## File Structure (Target)

```
ADVANCED_WEBSITE_DESIGN/
├── prisma/
│   └── schema.prisma          ✅ Created
├── database/
│   └── schema.sql             ✅ Created
├── src/
│   ├── lib/
│   │   ├── prisma.ts          🔄 Need to update
│   │   ├── supabase.ts        🔄 Need to update
│   │   └── auth.ts            ❌ Need to create
│   ├── app/
│   │   ├── login/
│   │   │   └── page.tsx       🔄 Need to update
│   │   ├── ai-workspace/
│   │   │   ├── page.tsx       🔄 Need to update
│   │   │   ├── layout.tsx     ✅ Done
│   │   │   └── components/    ✅ Done
│   │   └── api/
│   │       └── ai/
│   │           └── generate/
│   │               └── route.ts  🔄 Need to update
│   └── middleware.ts          ❌ Need to create
└── .env.local                 🔄 Need to update
```

---

## Key Changes Needed

### 1. Stop Code in Chat
**File**: `src/app/ai-workspace/page.tsx`
```typescript
// REMOVE: Displaying AI code in chat
// KEEP: Only status messages

// When AI streams code:
case 'ai-stream':
  // DON'T add to chat messages
  // ONLY update generatedFiles state
  const parsedFiles = parseFilesFromContent(fullResponse);
  setGeneratedFiles(parsedFiles);
  // NO setStreamingContent() for code
  break;
```

### 2. Save Messages to Database
**File**: `src/app/api/ai/generate/route.ts`
```typescript
import { prisma } from '@/lib/prisma';

// After AI completes:
const message = await prisma.message.create({
  data: {
    content: fullResponse,
    role: 'ASSISTANT',
    type: 'RESULT',
    projectId: projectId,
    Fragment: {
      create: {
        sandboxUrl: deploymentUrl,
        title: 'Generated Model',
        files: files,
      }
    }
  }
});
```

### 3. Implement Real Auth
**File**: `src/app/login/page.tsx`
```typescript
import { supabase } from '@/lib/supabase';

const handleLogin = async () => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  });
  
  if (!error) {
    router.push('/ai-workspace');
  }
};
```

### 4. Protect Routes
**File**: `src/middleware.ts`
```typescript
import { createMiddlewareClient } from '@supabase/auth-helpers-nextjs';

export async function middleware(req: NextRequest) {
  const supabase = createMiddlewareClient({ req, res });
  const { data: { session } } = await supabase.auth.getSession();
  
  if (!session && req.nextUrl.pathname.startsWith('/ai-workspace')) {
    return NextResponse.redirect(new URL('/login', req.url));
  }
}
```

---

## Environment Variables Needed

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
DATABASE_URL=postgresql://...

# AI APIs
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key

# E2B
E2B_API_KEY=your_key
```

---

## Testing Checklist

- [ ] User can sign up with email/password
- [ ] User can login with email/password
- [ ] User can login with Google OAuth
- [ ] Protected routes redirect to login
- [ ] Messages save to database
- [ ] Projects save to database
- [ ] Fragments save with sandbox URLs
- [ ] Code appears ONLY in Code tab
- [ ] Chat shows ONLY status messages
- [ ] E2B sandbox creates successfully
- [ ] Files are written to sandbox
- [ ] Training runs in sandbox
- [ ] API deploys successfully
- [ ] Sandbox URL saves to fragment
- [ ] User can view chat history
- [ ] User can create new projects
- [ ] User can switch between projects

---

## Next Steps

1. **Wait for Prisma installation** to complete
2. **Run** `npx prisma generate`
3. **Apply schema** to Supabase database
4. **Update** authentication implementation
5. **Update** API route to save messages
6. **Update** UI to remove code from chat
7. **Test** end-to-end flow

---

## Priority Order

1. 🔴 **HIGH**: Stop code from appearing in chat
2. 🔴 **HIGH**: Implement real authentication
3. 🟡 **MEDIUM**: Save messages to database
4. 🟡 **MEDIUM**: Save fragments with sandbox results
5. 🟢 **LOW**: Add chat history sidebar
6. 🟢 **LOW**: Add project switcher

---

## Estimated Time

- Phase 1 (Database): 30 minutes ✅
- Phase 2 (Auth): 1 hour
- Phase 3 (Messages): 1 hour
- Phase 4 (UI): 30 minutes
- Phase 5 (E2B): 30 minutes

**Total**: ~3.5 hours

---

## Files to Create/Update

### Create:
- [x] `prisma/schema.prisma`
- [x] `database/schema.sql`
- [ ] `src/middleware.ts`
- [ ] `src/lib/auth.ts`

### Update:
- [ ] `src/lib/prisma.ts`
- [ ] `src/lib/supabase.ts`
- [ ] `src/app/login/page.tsx`
- [ ] `src/app/ai-workspace/page.tsx`
- [ ] `src/app/api/ai/generate/route.ts`
- [ ] `.env.local`
- [ ] `.env.example`

---

## Current Status

✅ Database schema created
✅ Prisma schema created
🔄 Installing Prisma packages
⏳ Waiting to generate Prisma client
⏳ Waiting to implement auth
⏳ Waiting to update UI
⏳ Waiting to save messages

**Ready to proceed once Prisma installation completes!**
