# Fix: Training Jobs RLS Error

## Problem
When clicking the trigger button, you get this error:
```
Training start error: {
  code: '42501',
  details: null,
  hint: null,
  message: 'new row violates row-level security policy for table "training_jobs"'
}
```

## Root Cause
The `training_jobs` table has Row-Level Security (RLS) enabled in Supabase, but there are no policies configured to allow users to insert training jobs. This is a security feature that prevents unauthorized access.

## Solution

### Step 1: Go to Supabase Dashboard
1. Open https://supabase.com/dashboard
2. Select your project
3. Go to **SQL Editor** in the left sidebar

### Step 2: Run the RLS Policy Script
Copy and paste the entire content from `database/fix_training_jobs_rls.sql` into the SQL Editor and click **Run**.

This script will:
- Enable RLS on `training_jobs` table
- Add policies allowing users to:
  - View their own training jobs
  - Create training jobs for their own models
  - Update their own training jobs
  - Delete their own training jobs
- Also enable RLS on `ai_models` table with similar policies

### Step 3: Verify the Fix
After running the script, try clicking the trigger button again. The training should now start successfully.

## What the Policies Do

### For training_jobs table:
- **SELECT**: Users can only see training jobs where `user_id` matches their authenticated ID
- **INSERT**: Users can only create training jobs if:
  - The `user_id` matches their authenticated ID
  - The model belongs to them (checked via `ai_models` table)
- **UPDATE**: Users can only update their own training jobs
- **DELETE**: Users can only delete their own training jobs

### For ai_models table:
- **SELECT**: Users can only see their own models
- **INSERT**: Users can only create models for themselves
- **UPDATE**: Users can only update their own models
- **DELETE**: Users can only delete their own models

## Security Note
These policies ensure that:
- Users can only access their own data
- Users cannot access or modify other users' training jobs or models
- The backend API respects user boundaries

## If You Still Get Errors
1. Make sure you're logged in to Supabase with the correct account
2. Verify the script ran without errors (check for red error messages)
3. Try refreshing the page and attempting the trigger again
4. Check the browser console (F12) for more detailed error messages

## Alternative: Disable RLS (Not Recommended)
If you want to temporarily disable RLS for testing:
```sql
ALTER TABLE public.training_jobs DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_models DISABLE ROW LEVEL SECURITY;
```

However, this is **NOT RECOMMENDED** for production as it removes security restrictions.
