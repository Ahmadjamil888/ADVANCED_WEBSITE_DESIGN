# Setup: Add Supabase Service Role Key

## What is a Service Role Key?
A service role key is a special authentication key that allows backend operations to bypass Row-Level Security (RLS) policies. This is needed for the training API to create training jobs.

## Steps to Get Your Service Role Key

### Step 1: Go to Supabase Dashboard
1. Open https://supabase.com/dashboard
2. Select your project

### Step 2: Find the Service Role Key
1. Click **Settings** in the left sidebar
2. Click **API** in the submenu
3. Look for **Service Role** section
4. You'll see two keys:
   - `service_role` (this is what you need)
   - `anon` (this is the public key, don't use this)

### Step 3: Copy the Service Role Key
1. Click the **Copy** button next to the `service_role` key
2. Keep this key safe - it has full database access!

### Step 4: Add to .env.local
1. Open `.env.local` in your project root
2. Add this line:
```
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

Replace `your_service_role_key_here` with the key you copied in Step 3.

### Step 5: Restart Your Development Server
1. Stop your dev server (Ctrl+C)
2. Run it again: `npm run dev`
3. The new environment variable will be loaded

## Example .env.local
```
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
E2B_API_KEY=your_e2b_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
PAGE_ACCESS_PASSWORD=Ahmadjamil*25
```

## Security Warning ⚠️
- **Never commit** `.env.local` to git (it's in `.gitignore`)
- **Never share** your service role key publicly
- The service role key has full database access - treat it like a password
- Only use it on the backend, never expose it to the client

## Testing
After adding the key and restarting:
1. Go back to your app
2. Click the trigger button
3. Training should now work!

## Troubleshooting
- **"Service role not configured"**: Make sure you added the key to `.env.local` and restarted the server
- **Still getting RLS errors**: Make sure the key is correct and the dev server was restarted
