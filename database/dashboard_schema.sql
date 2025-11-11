-- Complete Dashboard Schema for AI Model Training Platform
-- This includes all tables needed for the full dashboard functionality

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS public.users (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  email character varying NOT NULL UNIQUE,
  username character varying NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  subscription_plan character varying DEFAULT 'free' CHECK (subscription_plan IN ('free', 'pro', 'enterprise')),
  subscription_expires_at timestamp with time zone,
  metadata jsonb DEFAULT '{}'::jsonb
);

-- AI Models table (main table for user's trained models)
CREATE TABLE IF NOT EXISTS public.ai_models (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  model_type character varying NOT NULL,
  framework character varying NOT NULL,
  base_model character varying,
  dataset_source character varying,
  dataset_name character varying,
  training_mode character varying DEFAULT 'from_scratch' CHECK (training_mode IN ('from_scratch', 'fine_tune')),
  training_status character varying DEFAULT 'pending' CHECK (training_status IN ('pending', 'queued', 'training', 'completed', 'failed', 'deployed')),
  huggingface_repo character varying,
  kaggle_dataset character varying,
  model_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  training_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  performance_metrics jsonb DEFAULT '{}'::jsonb,
  file_structure jsonb DEFAULT '{}'::jsonb,
  model_file_path text,
  model_file_format character varying CHECK (model_file_format IN ('pth', 'h5', 'pb', 'onnx', 'safetensors')),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deployed_at timestamp with time zone,
  deployment_type character varying DEFAULT 'none' CHECK (deployment_type IN ('local', 'e2b', 'aws', 'none')),
  deployment_url text,
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Training Jobs table
CREATE TABLE IF NOT EXISTS public.training_jobs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  model_id uuid NOT NULL REFERENCES public.ai_models(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  job_status character varying DEFAULT 'queued' CHECK (job_status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
  progress_percentage integer DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
  current_epoch integer DEFAULT 0,
  total_epochs integer DEFAULT 10,
  loss_value numeric,
  accuracy numeric,
  validation_loss numeric,
  validation_accuracy numeric,
  sandbox_session_id character varying,
  logs text,
  error_message text,
  started_at timestamp with time zone,
  completed_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Training Epochs (for real-time stats streaming)
CREATE TABLE IF NOT EXISTS public.training_epochs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  training_job_id uuid NOT NULL REFERENCES public.training_jobs(id) ON DELETE CASCADE,
  epoch_number integer NOT NULL,
  loss numeric,
  accuracy numeric,
  validation_loss numeric,
  validation_accuracy numeric,
  learning_rate numeric,
  timestamp timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- User Uploaded Datasets
CREATE TABLE IF NOT EXISTS public.user_datasets (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  file_path text NOT NULL,
  file_size bigint,
  file_type character varying,
  source_type character varying DEFAULT 'upload' CHECK (source_type IN ('upload', 'huggingface', 'kaggle')),
  source_url text,
  row_count integer,
  column_count integer,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- User Uploaded Models (for fine-tuning)
CREATE TABLE IF NOT EXISTS public.user_models (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  file_path text NOT NULL,
  file_size bigint,
  file_format character varying CHECK (file_format IN ('pth', 'h5', 'pb', 'onnx', 'safetensors')),
  framework character varying,
  base_model_name character varying,
  source_type character varying DEFAULT 'upload' CHECK (source_type IN ('upload', 'huggingface')),
  source_url text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Billing table
CREATE TABLE IF NOT EXISTS public.billing (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  plan_type character varying NOT NULL CHECK (plan_type IN ('free', 'pro', 'enterprise')),
  models_created integer DEFAULT 0,
  models_limit integer NOT NULL,
  has_api_access boolean DEFAULT false,
  billing_cycle_start date NOT NULL,
  billing_cycle_end date NOT NULL,
  is_paid boolean DEFAULT false,
  payment_method character varying,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- User Integrations (AWS, etc.)
CREATE TABLE IF NOT EXISTS public.user_integrations (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  service_name character varying NOT NULL CHECK (service_name IN ('aws', 'gcp', 'azure', 'huggingface', 'kaggle')),
  encrypted_api_key text,
  encrypted_secret text,
  is_active boolean DEFAULT true,
  last_used_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_ai_models_user_id ON public.ai_models(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_status ON public.ai_models(training_status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON public.training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON public.training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON public.training_jobs(job_status);
CREATE INDEX IF NOT EXISTS idx_training_epochs_job_id ON public.training_epochs(training_job_id);
CREATE INDEX IF NOT EXISTS idx_user_datasets_user_id ON public.user_datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_user_models_user_id ON public.user_models(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_user_id ON public.billing(user_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON public.ai_models
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON public.training_jobs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_billing_updated_at BEFORE UPDATE ON public.billing
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initialize billing for new users (default: free plan with 1 model)
CREATE OR REPLACE FUNCTION initialize_user_billing()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.billing (user_id, plan_type, models_limit, has_api_access, billing_cycle_start, billing_cycle_end)
  VALUES (NEW.id, 'free', 1, false, CURRENT_DATE, CURRENT_DATE + INTERVAL '1 month')
  ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER on_user_created AFTER INSERT ON public.users
FOR EACH ROW EXECUTE FUNCTION initialize_user_billing();

