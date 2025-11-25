-- API Keys table for user API key management
-- Run this SQL in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    key_hash TEXT NOT NULL, -- Hashed version of the API key
    key_preview VARCHAR(50) NOT NULL, -- First few characters for display
    is_active BOOLEAN DEFAULT TRUE,
    is_revoked BOOLEAN DEFAULT FALSE,
    max_daily_requests INTEGER DEFAULT 1000,
    max_monthly_requests INTEGER DEFAULT 10000,
    max_tokens_per_request INTEGER DEFAULT 2048,
    total_usage INTEGER DEFAULT 0,
    daily_usage INTEGER DEFAULT 0,
    monthly_usage INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active, is_revoked);

-- RLS (Row Level Security) Policy
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Policies for api_keys
CREATE POLICY "Users can manage their own API keys" ON api_keys FOR ALL USING (auth.uid() = user_id);

-- Trigger for automatic timestamp updates
CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();