-- ============================================
-- FIX: Add RLS Policies for training_jobs table
-- ============================================
-- Run this in Supabase SQL Editor to fix the RLS error

-- Enable RLS on training_jobs table
ALTER TABLE public.training_jobs ENABLE ROW LEVEL SECURITY;

-- Allow users to view their own training jobs
CREATE POLICY "Users can view own training jobs" ON public.training_jobs
  FOR SELECT USING (auth.uid() = user_id);

-- Allow users to create training jobs for their own models
CREATE POLICY "Users can create training jobs for own models" ON public.training_jobs
  FOR INSERT WITH CHECK (
    auth.uid() = user_id AND
    EXISTS (
      SELECT 1 FROM public.ai_models
      WHERE ai_models.id = training_jobs.model_id
      AND ai_models.user_id = auth.uid()
    )
  );

-- Allow users to update their own training jobs
CREATE POLICY "Users can update own training jobs" ON public.training_jobs
  FOR UPDATE USING (auth.uid() = user_id);

-- Allow users to delete their own training jobs
CREATE POLICY "Users can delete own training jobs" ON public.training_jobs
  FOR DELETE USING (auth.uid() = user_id);

-- ============================================
-- Also enable RLS on related tables
-- ============================================

ALTER TABLE public.ai_models ENABLE ROW LEVEL SECURITY;

-- Allow users to view their own models
CREATE POLICY "Users can view own models" ON public.ai_models
  FOR SELECT USING (auth.uid() = user_id);

-- Allow users to create models
CREATE POLICY "Users can create models" ON public.ai_models
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Allow users to update their own models
CREATE POLICY "Users can update own models" ON public.ai_models
  FOR UPDATE USING (auth.uid() = user_id);

-- Allow users to delete their own models
CREATE POLICY "Users can delete own models" ON public.ai_models
  FOR DELETE USING (auth.uid() = user_id);
