# ⚡ QUICK SETUP - 5 MINUTES

## 🚀 Get Zehanx AI Running in 5 Minutes

### 1. Install Dependencies (1 min)
```bash
cd ADVANCED_WEBSITE_DESIGN
pnpm install
```

### 2. Configure Environment (1 min)
Create `.env.local`:
```env
GROQ_API_KEY=your_groq_key
E2B_API_KEY=your_e2b_key
FIRECRAWL_API_KEY=your_firecrawl_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key
CLERK_SECRET_KEY=your_clerk_secret
```

### 3. Setup Database (1 min)
1. Go to Supabase dashboard
2. Open SQL Editor
3. Copy contents of `supabase_schema.sql`
4. Paste and execute

### 4. Start Server (1 min)
```bash
pnpm dev
```

### 5. Test (1 min)
1. Visit http://localhost:3000
2. Click "Try Our AI"
3. Login
4. Enter prompt: "Create sentiment analysis model"
5. Click Generate
6. Watch training happen!

---

## ✅ WHAT YOU GET

✅ **Prompt Box** - Enter any AI model description
✅ **Groq Integration** - Auto-generates training code
✅ **E2B Sandbox** - Executes code safely
✅ **Real-time Stats** - Live training progress
✅ **Supabase Storage** - Models saved to database
✅ **Model Download** - Download trained models
✅ **Model Management** - View, delete, organize

---

## 🎯 COMPLETE WORKFLOW

```
User Prompt
    ↓
Firecrawl Crawls Data
    ↓
Groq Generates Code
    ↓
E2B Executes Training
    ↓
Real-time Stats Display
    ↓
Model Saved to Supabase
    ↓
User Downloads Model
```

---

## 📋 FILES TO KNOW

| File | Purpose |
|------|---------|
| `src/app/api/train-model/route.ts` | Main training API (Groq + E2B + Supabase) |
| `src/app/api/models/route.ts` | List models from Supabase |
| `supabase_schema.sql` | Database schema |
| `COMPLETE_IMPLEMENTATION_GUIDE.md` | Full setup guide |
| `FINAL_STATUS_REPORT.md` | What's implemented |

---

## 🆘 COMMON ISSUES

**Issue**: "API key not configured"
**Fix**: Add to `.env.local` and restart server

**Issue**: "Supabase connection failed"
**Fix**: Run `supabase_schema.sql` in Supabase dashboard

**Issue**: "Training fails"
**Fix**: Check all API keys are valid

---

## 🚀 DEPLOY TO PRODUCTION

```bash
vercel deploy
# Add environment variables in Vercel dashboard
```

---

**That's it! You're ready to go!** 🎉

For detailed info, see `COMPLETE_IMPLEMENTATION_GUIDE.md`
