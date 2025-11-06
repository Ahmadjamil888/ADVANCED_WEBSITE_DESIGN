import { createClient } from '@supabase/supabase-js'

export type Database = {
  public: {
    Tables: {
      messages: {
        Row: {
          id: string
          chat_id: string
          role: string
          content: string
          metadata?: {
            deploymentUrl?: string
            downloadUrl?: string
            files?: string[]
            type?: string
            modelType?: string
            baseModel?: string
            modelId?: string
          } | null
          created_at: string
        }
        Insert: {
          id?: string
          chat_id: string
          role: string
          content: string
          metadata?: {
            deploymentUrl?: string
            downloadUrl?: string
            files?: string[]
            type?: string
            modelType?: string
            baseModel?: string
            modelId?: string
          } | null
          created_at?: string
        }
        Update: {
          id?: string
          chat_id?: string
          role?: string
          content?: string
          metadata?: {
            deploymentUrl?: string
            downloadUrl?: string
            files?: string[]
            type?: string
            modelType?: string
            baseModel?: string
            modelId?: string
          } | null
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
          metadata: any
        }
        Insert: {
          id?: string
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
          metadata?: any
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
          metadata?: any
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
          metadata: any
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
          metadata?: any
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
          metadata?: any
        }
      }
    }
  }
}

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''

export const supabase = (() => {
  if (!supabaseUrl || !supabaseAnonKey) {
    console.error('Missing Supabase URL or Anon Key')
    return null
  }
  
  return createClient<Database>(supabaseUrl, supabaseAnonKey, {
    db: {
      schema: 'public'
    },
    auth: {
      persistSession: true
    }
  })
})()