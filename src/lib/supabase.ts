import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

export type Database = {
  public: {
    Tables: {
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