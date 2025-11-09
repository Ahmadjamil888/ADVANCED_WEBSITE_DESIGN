import { createClient } from '@supabase/supabase-js';
import type { Database } from './database.types';

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
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error("❌ Missing Supabase environment variables in .env.local");
}

// Create the Supabase client
export const supabase = createClient<Database>(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  {
    db: { schema: 'public' },
    auth: { persistSession: true },
  }
);
