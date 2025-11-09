// Generated from Supabase schema

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      ai_models: {
        Row: {
          id: string
          user_id: string
          name: string
          description: string | null
          model_type: string
          framework: string
          base_model: string | null
          dataset_source: string | null
          dataset_name: string | null
          training_status: string
          huggingface_repo: string | null
          model_config: Json
          training_config: Json
          performance_metrics: Json | null
          file_structure: Json | null
          metadata: Json | null
          created_at: string | null
          updated_at: string | null
          deployed_at: string | null
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          description?: string | null
          model_type: string
          framework: string
          base_model?: string | null
          dataset_source?: string | null
          dataset_name?: string | null
          training_status?: string
          huggingface_repo?: string | null
          model_config?: Json
          training_config?: Json
          performance_metrics?: Json | null
          file_structure?: Json | null
          metadata?: Json | null
          created_at?: string | null
          updated_at?: string | null
          deployed_at?: string | null
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          description?: string | null
          model_type?: string
          framework?: string
          base_model?: string | null
          dataset_source?: string | null
          dataset_name?: string | null
          training_status?: string
          huggingface_repo?: string | null
          model_config?: Json
          training_config?: Json
          performance_metrics?: Json | null
          file_structure?: Json | null
          metadata?: Json | null
          created_at?: string | null
          updated_at?: string | null
          deployed_at?: string | null
        }
      }
      chats: {
        Row: {
          id: string
          user_id: string
          title: string
          mode: string
          created_at: string
          updated_at: string
          is_pinned: boolean
          last_message: string | null
        }
        Insert: {
          id?: string
          user_id: string
          title: string
          mode?: string
          created_at?: string
          updated_at?: string
          is_pinned?: boolean
          last_message?: string | null
        }
        Update: {
          id?: string
          user_id?: string
          title?: string
          mode?: string
          created_at?: string
          updated_at?: string
          is_pinned?: boolean
          last_message?: string | null
        }
      }
      messages: {
        Row: {
          id: string
          chat_id: string
          role: 'user' | 'assistant' | 'system'
          content: string
          model_used: string | null
          metadata: Json | null
          created_at: string
        }
        Insert: {
          id?: string
          chat_id: string
          role: 'user' | 'assistant' | 'system'
          content: string
          model_used?: string | null
          metadata?: Json | null
          created_at?: string
        }
        Update: {
          id?: string
          chat_id?: string
          role?: 'user' | 'assistant' | 'system'
          content?: string
          model_used?: string | null
          metadata?: Json | null
          created_at?: string
        }
      }
      users: {
        Row: {
          id: string
          email: string
          username: string
          created_at: string
          updated_at: string
          total_tokens_used: number
          total_requests: number
          last_activity: string | null
          is_active: boolean
          is_premium: boolean
          daily_requests: number
          monthly_requests: number
          last_reset_date: string
          metadata: Json | null
        }
        Insert: {
          id: string
          email: string
          username: string
          created_at?: string
          updated_at?: string
          total_tokens_used?: number
          total_requests?: number
          last_activity?: string | null
          is_active?: boolean
          is_premium?: boolean
          daily_requests?: number
          monthly_requests?: number
          last_reset_date?: string
          metadata?: Json | null
        }
        Update: {
          id?: string
          email?: string
          username?: string
          created_at?: string
          updated_at?: string
          total_tokens_used?: number
          total_requests?: number
          last_activity?: string | null
          is_active?: boolean
          is_premium?: boolean
          daily_requests?: number
          monthly_requests?: number
          last_reset_date?: string
          metadata?: Json | null
        }
      }
      model_usage: {
        Row: {
          id: string
          user_id: string
          model_name: string
          model_version: string | null
          prompt_tokens: number
          completion_tokens: number
          total_tokens: number
          temperature: number | null
          top_p: number | null
          max_tokens: number | null
          cost_credits: number
          created_at: string
          metadata: Json | null
        }
        Insert: {
          id?: string
          user_id: string
          model_name: string
          model_version?: string | null
          prompt_tokens: number
          completion_tokens: number
          total_tokens: number
          temperature?: number | null
          top_p?: number | null
          max_tokens?: number | null
          cost_credits?: number
          created_at?: string
          metadata?: Json | null
        }
        Update: {
          id?: string
          user_id?: string
          model_name?: string
          model_version?: string | null
          prompt_tokens?: number
          completion_tokens?: number
          total_tokens?: number
          temperature?: number | null
          top_p?: number | null
          max_tokens?: number | null
          cost_credits?: number
          created_at?: string
          metadata?: Json | null
        }
      }
      user_integrations: {
        Row: {
          id: string
          user_id: string
          service_name: string
          encrypted_api_key: string
          is_active: boolean
          last_used_at: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          service_name: string
          encrypted_api_key: string
          is_active?: boolean
          last_used_at?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          service_name?: string
          encrypted_api_key?: string
          is_active?: boolean
          last_used_at?: string
          created_at?: string
          updated_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}
