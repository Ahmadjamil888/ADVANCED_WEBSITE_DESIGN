-- Zehanx AI Supabase Schema
-- Run this SQL in your Supabase dashboard

-- Create trained_models table
CREATE TABLE IF NOT EXISTS trained_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  model_type TEXT NOT NULL CHECK (model_type IN ('transformer', 'lstm', 'cnn', 'custom')),
  dataset_source TEXT NOT NULL CHECK (dataset_source IN ('firecrawl', 'huggingface', 'kaggle', 'github')),
  final_loss FLOAT,
  final_accuracy FLOAT,
  epochs_trained INTEGER DEFAULT 0,
  model_path TEXT,
  stats JSONB DEFAULT '[]'::jsonb,
  sandbox_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create training_jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  model_id UUID REFERENCES trained_models(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'initializing', 'running', 'completed', 'failed')),
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  current_epoch INTEGER DEFAULT 0,
  total_epochs INTEGER DEFAULT 0,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trained_models_user_id ON trained_models(user_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_created_at ON trained_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);

-- Enable Row Level Security (RLS)
ALTER TABLE trained_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for trained_models
CREATE POLICY "Users can view their own trained models"
  ON trained_models FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own trained models"
  ON trained_models FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own trained models"
  ON trained_models FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own trained models"
  ON trained_models FOR DELETE
  USING (auth.uid() = user_id);

-- Create RLS policies for training_jobs
CREATE POLICY "Users can view their own training jobs"
  ON training_jobs FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own training jobs"
  ON training_jobs FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own training jobs"
  ON training_jobs FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own training jobs"
  ON training_jobs FOR DELETE
  USING (auth.uid() = user_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_trained_models_updated_at
  BEFORE UPDATE ON trained_models
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at
  BEFORE UPDATE ON training_jobs
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON trained_models TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON training_jobs TO authenticated;
