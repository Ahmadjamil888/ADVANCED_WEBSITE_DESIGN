// Supabase Database Client
// Direct database operations using Supabase

import { createClient } from '@supabase/supabase-js';

if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
  throw new Error('Missing NEXT_PUBLIC_SUPABASE_URL');
}

if (!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
  throw new Error('Missing NEXT_PUBLIC_SUPABASE_ANON_KEY');
}

// Create Supabase client
export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

// Database helper functions
export const db = {
  // Chats
  async createChat(userId: string, title?: string) {
    const { data, error } = await supabase
      .from('chats')
      .insert({
        user_id: userId,
        title: title || 'Untitled Chat',
      })
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  async getChats(userId: string) {
    const { data, error } = await supabase
      .from('chats')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });
    
    if (error) throw error;
    return data;
  },

  async getChat(chatId: string) {
    const { data, error } = await supabase
      .from('chats')
      .select('*')
      .eq('id', chatId)
      .single();
    
    if (error) throw error;
    return data;
  },

  // Messages
  async createMessage(chatId: string, role: string, content: string, metadata?: any) {
    const { data, error } = await supabase
      .from('messages')
      .insert({
        chat_id: chatId,
        role,
        content,
        metadata: metadata || {},
      })
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  async getMessages(chatId: string) {
    const { data, error} = await supabase
      .from('messages')
      .select('*, fragments(*)')
      .eq('chat_id', chatId)
      .order('created_at', { ascending: true });
    
    if (error) throw error;
    return data;
  },

  // Fragments
  async createFragment(messageId: string, sandboxUrl: string, sandboxId: string, title: string, files: any) {
    const { data, error } = await supabase
      .from('fragments')
      .insert({
        message_id: messageId,
        sandbox_url: sandboxUrl,
        sandbox_id: sandboxId,
        title,
        files,
      })
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  // Users
  async getUser(userId: string) {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .eq('id', userId)
      .single();
    
    if (error) throw error;
    return data;
  },

  async updateUser(userId: string, updates: any) {
    const { data, error } = await supabase
      .from('users')
      .update(updates)
      .eq('id', userId)
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },
};

// TypeScript types for database tables
export type Chat = {
  id: string;
  user_id: string;
  title: string;
  mode: string;
  model_name: string;
  temperature: number;
  max_tokens: number;
  system_prompt: string | null;
  is_pinned: boolean;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
  metadata: any;
};

export type Message = {
  id: string;
  chat_id: string;
  role: string;
  content: string;
  tokens_used: number;
  model_used: string | null;
  processing_time_ms: number | null;
  created_at: string;
  metadata: any;
  fragments?: Fragment;
};

export type Fragment = {
  id: string;
  message_id: string;
  sandbox_url: string;
  sandbox_id: string | null;
  title: string;
  files: any;
  created_at: string;
  updated_at: string;
};

export type User = {
  id: string;
  email: string;
  username: string;
  created_at: string;
  updated_at: string;
  total_tokens_used: number;
  total_requests: number;
  last_activity: string | null;
  is_active: boolean;
  is_premium: boolean;
  daily_requests: number;
  monthly_requests: number;
  last_reset_date: string;
  metadata: any;
};
