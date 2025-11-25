-- ============================================
-- AI WORKSPACE - Complete Supabase Schema
-- ============================================
-- Run this in Supabase SQL Editor
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- ENUMS
-- ============================================

-- Message Role Enum
DO $$ BEGIN
    CREATE TYPE message_role AS ENUM ('USER', 'ASSISTANT');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Message Type Enum
DO $$ BEGIN
    CREATE TYPE message_type AS ENUM ('RESULT', 'ERROR');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================
-- USERS TABLE (extends auth.users)
-- ============================================

CREATE TABLE IF NOT EXISTS public.users (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
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
  metadata jsonb DEFAULT '{}'::jsonb
);

-- ============================================
-- CHATS TABLE (Projects/Workspaces)
-- ============================================

CREATE TABLE IF NOT EXISTS public.chats (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
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

-- ============================================
-- MESSAGES TABLE
-- ============================================

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

-- ============================================
-- FRAGMENTS TABLE (Sandbox Results)
-- ============================================

CREATE TABLE IF NOT EXISTS public.fragments (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  message_id uuid NOT NULL UNIQUE REFERENCES public.messages(id) ON DELETE CASCADE,
  sandbox_url text NOT NULL,
  sandbox_id text,
  title text NOT NULL,
  files jsonb NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

-- ============================================
-- AI MODELS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.ai_models (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  model_type character varying NOT NULL,
  framework character varying NOT NULL,
  base_model character varying,
  dataset_source character varying,
  dataset_name character varying,
  training_status character varying DEFAULT 'pending',
  huggingface_repo character varying,
  model_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  training_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  performance_metrics jsonb DEFAULT '{}'::jsonb,
  file_structure jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  deployed_at timestamp with time zone,
  metadata jsonb DEFAULT '{}'::jsonb
);

-- ============================================
-- TRAINING JOBS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.training_jobs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  model_id uuid NOT NULL REFERENCES public.ai_models(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  job_status character varying DEFAULT 'queued',
  progress_percentage integer DEFAULT 0,
  current_epoch integer DEFAULT 0,
  total_epochs integer DEFAULT 10,
  loss_value numeric,
  accuracy numeric,
  sandbox_session_id character varying,
  logs text,
  error_message text,
  started_at timestamp with time zone,
  completed_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- ============================================
-- API KEYS TABLE
-- ============================================

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

-- ============================================
-- ADDITIONAL TABLES
-- ============================================

CREATE TABLE IF NOT EXISTS public.billing (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  credits_balance integer NOT NULL DEFAULT 0,
  credits_spent integer NOT NULL DEFAULT 0,
  billing_cycle_start date NOT NULL,
  billing_cycle_end date NOT NULL,
  is_paid boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.chat_entities (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  chat_id uuid NOT NULL REFERENCES public.chats(id) ON DELETE CASCADE,
  entity_type character varying NOT NULL,
  entity_name character varying NOT NULL,
  entity_value text,
  confidence_score numeric DEFAULT 1.0,
  first_mentioned_at timestamp with time zone DEFAULT now(),
  last_mentioned_at timestamp with time zone DEFAULT now(),
  mention_count integer DEFAULT 1,
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.chat_files (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  chat_id uuid NOT NULL REFERENCES public.chats(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  filename character varying NOT NULL,
  file_type character varying NOT NULL,
  file_size integer NOT NULL,
  file_url text,
  content text,
  is_indexed boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.generated_apps (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  chat_id uuid REFERENCES public.chats(id) ON DELETE SET NULL,
  name character varying NOT NULL,
  description text,
  app_type character varying NOT NULL,
  framework character varying NOT NULL,
  deployment_status character varying DEFAULT 'draft',
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

CREATE TABLE IF NOT EXISTS public.model_usage (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  api_key_id uuid REFERENCES public.api_keys(id) ON DELETE SET NULL,
  model_name character varying NOT NULL,
  model_version character varying,
  prompt_tokens integer NOT NULL,
  completion_tokens integer NOT NULL,
  total_tokens integer NOT NULL,
  temperature double precision,
  top_p double precision,
  max_tokens integer,
  cost_credits numeric DEFAULT 0,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.prompt_templates (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name character varying NOT NULL,
  description text,
  category character varying DEFAULT 'general',
  template_content text NOT NULL,
  variables jsonb DEFAULT '[]'::jsonb,
  is_public boolean DEFAULT false,
  usage_count integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.rate_limits (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  api_key_id uuid NOT NULL REFERENCES public.api_keys(id) ON DELETE CASCADE,
  window_start timestamp with time zone NOT NULL,
  window_end timestamp with time zone NOT NULL,
  request_count integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS public.usage_logs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  api_key_id uuid REFERENCES public.api_keys(id) ON DELETE SET NULL,
  endpoint character varying NOT NULL,
  method character varying NOT NULL,
  ip_address inet,
  user_agent text,
  tokens_used integer NOT NULL DEFAULT 0,
  response_time_ms integer,
  request_size_bytes integer,
  response_size_bytes integer,
  status_code integer,
  success boolean NOT NULL,
  error_message text,
  created_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.user_integrations (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  service_name character varying NOT NULL,
  encrypted_api_key text,
  is_active boolean DEFAULT true,
  last_used_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.user_sessions (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  session_token character varying NOT NULL UNIQUE,
  expires_at timestamp with time zone NOT NULL,
  device_info jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.user_tools (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  tool_name character varying NOT NULL,
  tool_type character varying NOT NULL,
  is_enabled boolean DEFAULT true,
  configuration jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now()
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX IF NOT EXISTS idx_chats_user_id ON public.chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_created_at ON public.chats(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON public.messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON public.messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fragments_message_id ON public.fragments(message_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_user_id ON public.ai_models(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON public.training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON public.training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_user_id ON public.usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON public.usage_logs(created_at DESC);

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fragments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view own profile" ON public.users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.users
  FOR UPDATE USING (auth.uid() = id);

-- Chats policies
CREATE POLICY "Users can view own chats" ON public.chats
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own chats" ON public.chats
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own chats" ON public.chats
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own chats" ON public.chats
  FOR DELETE USING (auth.uid() = user_id);

-- Messages policies
CREATE POLICY "Users can view messages in own chats" ON public.messages
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.chats
      WHERE chats.id = messages.chat_id
      AND chats.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create messages in own chats" ON public.messages
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.chats
      WHERE chats.id = messages.chat_id
      AND chats.user_id = auth.uid()
    )
  );

-- Fragments policies
CREATE POLICY "Users can view fragments in own chats" ON public.fragments
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.messages
      JOIN public.chats ON chats.id = messages.chat_id
      WHERE messages.id = fragments.message_id
      AND chats.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create fragments in own chats" ON public.fragments
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.messages
      JOIN public.chats ON chats.id = messages.chat_id
      WHERE messages.id = fragments.message_id
      AND chats.user_id = auth.uid()
    )
  );

-- AI Models policies
CREATE POLICY "Users can view own models" ON public.ai_models
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own models" ON public.ai_models
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own models" ON public.ai_models
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own models" ON public.ai_models
  FOR DELETE USING (auth.uid() = user_id);

-- Training Jobs policies
CREATE POLICY "Users can view own training jobs" ON public.training_jobs
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own training jobs" ON public.training_jobs
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- API Keys policies
CREATE POLICY "Users can view own API keys" ON public.api_keys
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create own API keys" ON public.api_keys
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own API keys" ON public.api_keys
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own API keys" ON public.api_keys
  FOR DELETE USING (auth.uid() = user_id);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chats_updated_at BEFORE UPDATE ON public.chats
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fragments_updated_at BEFORE UPDATE ON public.fragments
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON public.ai_models
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON public.api_keys
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.users (id, email, username)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1))
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create user profile on signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- COMPLETED!
-- ============================================
-- Schema created successfully
-- Next steps:
-- 1. Run this SQL in Supabase SQL Editor
-- 2. Enable Email Auth in Supabase Dashboard
-- 3. Configure OAuth providers if needed
-- 4. Update .env.local with Supabase credentials
-- ============================================
