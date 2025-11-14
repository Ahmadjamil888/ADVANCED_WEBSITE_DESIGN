-- ============================================
-- FIX: Drop and Recreate RLS Policies (v2)
-- ============================================
-- Run this in Supabase SQL Editor to fix the RLS error

-- Drop existing policies on training_jobs if they exist
DROP POLICY IF EXISTS "Users can view own training jobs" ON public.training_jobs;
DROP POLICY IF EXISTS "Users can create training jobs for own models" ON public.training_jobs;
DROP POLICY IF EXISTS "Users can update own training jobs" ON public.training_jobs;
DROP POLICY IF EXISTS "Users can delete own training jobs" ON public.training_jobs;

-- Drop existing policies on ai_models if they exist
DROP POLICY IF EXISTS "Users can view own models" ON public.ai_models;
DROP POLICY IF EXISTS "Users can create models" ON public.ai_models;
DROP POLICY IF EXISTS "Users can update own models" ON public.ai_models;
DROP POLICY IF EXISTS "Users can delete own models" ON public.ai_models;

-- Enable RLS on training_jobs table
ALTER TABLE public.training_jobs ENABLE ROW LEVEL SECURITY;

-- Create new policies for training_jobs
CREATE POLICY "Users can create training jobs for own models" ON public.training_jobs
  FOR INSERT WITH CHECK (
    auth.uid() = user_id AND
    EXISTS (
      SELECT 1 FROM public.ai_models
      WHERE ai_models.id = training_jobs.model_id
      AND ai_models.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can view own training jobs" ON public.training_jobs
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own training jobs" ON public.training_jobs
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own training jobs" ON public.training_jobs
  FOR DELETE USING (auth.uid() = user_id);

-- Enable RLS on ai_models table
ALTER TABLE public.ai_models ENABLE ROW LEVEL SECURITY;

-- Create new policies for ai_models
CREATE POLICY "Users can create models" ON public.ai_models
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own models" ON public.ai_models
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own models" ON public.ai_models
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own models" ON public.ai_models
  FOR DELETE USING (auth.uid() = user_id);
