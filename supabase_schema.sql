-- Zehanx AI Supabase Schema
-- Run this SQL in your Supabase dashboard

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table (extends auth.users)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create datasets table
CREATE TABLE IF NOT EXISTS datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  source TEXT NOT NULL CHECK (source IN ('huggingface', 'kaggle', 'firecrawl', 'upload', 'github')),
  source_url TEXT,
  size_bytes BIGINT,
  num_samples INTEGER,
  num_features INTEGER,
  data_types JSONB DEFAULT '{}',
  validation_status TEXT DEFAULT 'pending' CHECK (validation_status IN ('pending', 'validating', 'valid', 'invalid')),
  validation_errors JSONB DEFAULT '[]',
  preprocessing_status TEXT DEFAULT 'pending' CHECK (preprocessing_status IN ('pending', 'processing', 'completed', 'failed')),
  processed_path TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create models table
CREATE TABLE IF NOT EXISTS models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  architecture TEXT NOT NULL,
  framework TEXT NOT NULL DEFAULT 'transformers',
  base_model TEXT,
  config JSONB DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'designing' CHECK (status IN ('designing', 'training', 'trained', 'deployed', 'failed')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create training_runs table
CREATE TABLE IF NOT EXISTS training_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'initializing', 'running', 'completed', 'failed')),
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  current_epoch INTEGER DEFAULT 0,
  total_epochs INTEGER DEFAULT 0,
  batch_size INTEGER DEFAULT 32,
  learning_rate FLOAT DEFAULT 0.001,
  loss_function TEXT DEFAULT 'cross_entropy',
  optimizer TEXT DEFAULT 'adam',
  metrics JSONB DEFAULT '{}',
  logs JSONB DEFAULT '[]',
  model_path TEXT,
  error_message TEXT,
  gpu_used BOOLEAN DEFAULT FALSE,
  gpu_memory_used BIGINT,
  training_time_seconds INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create deployments table
CREATE TABLE IF NOT EXISTS deployments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  training_run_id UUID REFERENCES training_runs(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'deploying' CHECK (status IN ('deploying', 'running', 'stopped', 'failed')),
  endpoint_url TEXT,
  api_key TEXT,
  deployment_config JSONB DEFAULT '{}',
  performance_metrics JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  training_run_id UUID REFERENCES training_runs(id) ON DELETE CASCADE,
  deployment_id UUID REFERENCES deployments(id) ON DELETE CASCADE,
  metric_name TEXT NOT NULL,
  metric_value FLOAT NOT NULL,
  step INTEGER,
  epoch INTEGER,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trained_models table (legacy, keeping for compatibility)
CREATE TABLE IF NOT EXISTS trained_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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

-- Create training_jobs table (legacy, keeping for compatibility)
CREATE TABLE IF NOT EXISTS training_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(source);
CREATE INDEX IF NOT EXISTS idx_datasets_validation_status ON datasets(validation_status);
CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);
CREATE INDEX IF NOT EXISTS idx_models_project_id ON models(project_id);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_user_id ON training_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_model_id ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_dataset_id ON training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_deployments_user_id ON deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_deployments_model_id ON deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_metrics_training_run_id ON metrics(training_run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_deployment_id ON metrics(deployment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_trained_models_user_id ON trained_models(user_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_created_at ON trained_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);

-- Enable Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
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
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at
  BEFORE UPDATE ON datasets
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at
  BEFORE UPDATE ON models
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_runs_updated_at
  BEFORE UPDATE ON training_runs
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_deployments_updated_at
  BEFORE UPDATE ON deployments
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trained_models_updated_at
  BEFORE UPDATE ON trained_models
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at
  BEFORE UPDATE ON training_jobs
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON users TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON projects TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON datasets TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON models TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON training_runs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON deployments TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON metrics TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON trained_models TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON training_jobs TO authenticated;