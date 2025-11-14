import { createClient } from '@supabase/supabase-js';
import type { Database } from './database.types';
import type { SupabaseClient } from '@supabase/supabase-js';

export type { Database };

export type Tables = Database['public']['Tables'];
type TableName = keyof Tables;
type TableRow<T extends TableName> = Tables[T]['Row'];
type TableInsert<T extends TableName> = Tables[T]['Insert'];
type TableUpdate<T extends TableName> = Tables[T]['Update'];

/** -------------------------------
 * 🧠 AI Model Table Type
 * --------------------------------
 */
export type AIModel = {
  id?: string;
  user_id: string;
  name: string;
  description: string;
  model_type: string;
  framework: string;
  base_model: string;
  dataset_source: string;
  training_status: string;
  file_structure?: {
    files: string[];
  };
  metadata: {
    eventId: string;
    deploymentMethod: string;
    allFilesUploaded: boolean;
    [key: string]: any;
  };
  created_at?: string;
  updated_at?: string;
  deployed_at: string;
};

/** ✅ Load environment variables safely */
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

/**
 * Do not throw on missing env in the client bundle.
 * Many components guard against a null client; throwing here causes a client-side crash.
 */
export const supabase = (supabaseUrl && supabaseAnonKey)
  ? createClient<Database>(
      supabaseUrl,
      supabaseAnonKey,
      {
        db: { schema: 'public' },
        auth: { persistSession: true },
      }
    )
  : undefined;

/**
 * Service role client for backend operations that bypass RLS
 * Only use this on the server side for trusted operations
 */
export function getSupabaseServiceRole(): SupabaseClient<Database> {
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    throw new Error('Supabase service role is not configured. Set SUPABASE_SERVICE_ROLE_KEY in environment variables.');
  }
  return createClient<Database>(supabaseUrl, supabaseServiceRoleKey, {
    db: { schema: 'public' },
  });
}

/**
 * Returns a non-undefined Supabase client or throws with a clear error.
 * Use this inside functions to satisfy TypeScript and avoid crashing at import time.
 */
export function getSupabaseOrThrow(): SupabaseClient<Database> {
  if (!supabase) {
    throw new Error('Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.');
  }
  return supabase;
}
