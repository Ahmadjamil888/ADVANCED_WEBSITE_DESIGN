-- Complete Database Schema for AI Model Dashboard
-- Run this script to set up all required tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types
CREATE TYPE "MessageRole" AS ENUM ('USER', 'ASSISTANT');
CREATE TYPE "MessageType" AS ENUM ('RESULT', 'ERROR');

-- Users table (extends auth.users)
CREATE TABLE IF NOT EXISTS public.users (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  email character varying NOT NULL UNIQUE,
  username character varying NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  total_tokens_used integer DEFAULT 0,
  total_requests integer DEFAULT 0,
  last_activity timestamp with time zone,
  is_active boolean DEFAULT true,
  is_premium boolean DEFAULT false,
  daily_requests integer DEFAULT 0,
  monthly_requests integer DEFAULT 0,
  last_reset_date date DEFAULT CURRENT_DATE,
  subscription_plan character varying DEFAULT 'free' CHECK (subscription_plan IN ('free', 'pro', 'enterprise')),
  subscription_expires_at timestamp with time zone,
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Projects table (for organizing AI models)
CREATE TABLE IF NOT EXISTS public.projects (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- AI Models table
CREATE TABLE IF NOT EXISTS public.ai_models (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  project_id uuid REFERENCES public.projects(id) ON DELETE SET NULL,
  name character varying NOT NULL,
  description text,
  model_type character varying NOT NULL,
  framework character varying NOT NULL,
  base_model character varying,
  dataset_source character varying,
  dataset_name character varying,
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
  deployment_type character varying CHECK (deployment_type IN ('local', 'e2b', 'aws', 'none')),
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

-- Training Epochs (for real-time stats)
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
  row_count integer,
  column_count integer,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- User Uploaded Models
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
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Chats table
CREATE TABLE IF NOT EXISTS public.chats (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title character varying NOT NULL DEFAULT 'Untitled Chat',
  mode character varying NOT NULL DEFAULT 'chat',
  model_name character varying DEFAULT 'gemini-1.5-pro',
  temperature numeric DEFAULT 0.7,
  max_tokens integer DEFAULT 2048,
  system_prompt text,
  is_pinned boolean DEFAULT false,
  is_archived boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Messages table
CREATE TABLE IF NOT EXISTS public.messages (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  chat_id uuid NOT NULL REFERENCES public.chats(id) ON DELETE CASCADE,
  role character varying NOT NULL,
  content text NOT NULL,
  tokens_used integer DEFAULT 0,
  model_used character varying,
  processing_time_ms integer,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Billing table
CREATE TABLE IF NOT EXISTS public.billing (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  plan_type character varying NOT NULL CHECK (plan_type IN ('free', 'pro', 'enterprise')),
  credits_balance integer NOT NULL DEFAULT 0,
  credits_spent integer NOT NULL DEFAULT 0,
  models_created integer DEFAULT 0,
  models_limit integer NOT NULL,
  billing_cycle_start date NOT NULL,
  billing_cycle_end date NOT NULL,
  is_paid boolean DEFAULT false,
  payment_method character varying,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- API Keys table
CREATE TABLE IF NOT EXISTS public.api_keys (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  key_hash character varying NOT NULL UNIQUE,
  key_preview character varying NOT NULL,
  name character varying NOT NULL,
  description text,
  is_active boolean DEFAULT true,
  is_revoked boolean DEFAULT false,
  max_daily_requests integer DEFAULT 1000,
  max_monthly_requests integer DEFAULT 10000,
  max_tokens_per_request integer DEFAULT 2048,
  expires_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  total_usage integer DEFAULT 0,
  daily_usage integer DEFAULT 0,
  monthly_usage integer DEFAULT 0,
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

-- Generated Apps table
CREATE TABLE IF NOT EXISTS public.generated_apps (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  chat_id uuid REFERENCES public.chats(id) ON DELETE SET NULL,
  name character varying NOT NULL,
  description text,
  app_type character varying NOT NULL,
  framework character varying NOT NULL,
  deployment_status character varying DEFAULT 'draft' CHECK (deployment_status IN ('draft', 'deploying', 'deployed', 'failed')),
  deployment_url text,
  repository_url text,
  source_code jsonb NOT NULL DEFAULT '{}'::jsonb,
  dependencies jsonb DEFAULT '[]'::jsonb,
  environment_vars jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deployed_at timestamp with time zone,
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_ai_models_user_id ON public.ai_models(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_status ON public.ai_models(training_status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON public.training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON public.training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON public.training_jobs(job_status);
CREATE INDEX IF NOT EXISTS idx_training_epochs_job_id ON public.training_epochs(training_job_id);
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON public.chats(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON public.messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_billing_user_id ON public.billing(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_user_integrations_user_id ON public.user_integrations(user_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON public.ai_models
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON public.training_jobs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chats_updated_at BEFORE UPDATE ON public.chats
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_billing_updated_at BEFORE UPDATE ON public.billing
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initialize billing for existing users
INSERT INTO public.billing (user_id, plan_type, models_limit, billing_cycle_start, billing_cycle_end)
SELECT id, 'free', 1, CURRENT_DATE, CURRENT_DATE + INTERVAL '1 month'
FROM public.users
WHERE id NOT IN (SELECT user_id FROM public.billing)
ON CONFLICT DO NOTHING;

