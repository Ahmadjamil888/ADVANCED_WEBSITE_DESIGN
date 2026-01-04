// Supabase Database Client
import { createClient } from '@supabase/supabase-js';
import { config } from '../config';

export const supabase = createClient(
  config.supabase.url,
  config.supabase.anonKey,
  {
    auth: {
      autoRefreshToken: true,
      persistSession: true,
    },
  }
);

// Server-side client for API routes
export const supabaseServer = createClient(
  config.supabase.url,
  process.env.SUPABASE_SERVICE_ROLE_KEY || config.supabase.anonKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
);

// Database types (will be generated with supabase gen types)
export type Database = {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          email: string;
          name: string | null;
          avatar_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          email: string;
          name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string;
          name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      projects: {
        Row: {
          id: string;
          user_id: string;
          name: string;
          description: string | null;
          status: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          name: string;
          description?: string | null;
          status?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          name?: string;
          description?: string | null;
          status?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      datasets: {
        Row: {
          id: string;
          user_id: string;
          project_id: string | null;
          name: string;
          description: string | null;
          source: string;
          source_url: string | null;
          size_bytes: number | null;
          num_samples: number | null;
          num_features: number | null;
          data_types: any;
          validation_status: string;
          validation_errors: any;
          preprocessing_status: string;
          processed_path: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          project_id?: string | null;
          name: string;
          description?: string | null;
          source: string;
          source_url?: string | null;
          size_bytes?: number | null;
          num_samples?: number | null;
          num_features?: number | null;
          data_types?: any;
          validation_status?: string;
          validation_errors?: any;
          preprocessing_status?: string;
          processed_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          project_id?: string | null;
          name?: string;
          description?: string | null;
          source?: string;
          source_url?: string | null;
          size_bytes?: number | null;
          num_samples?: number | null;
          num_features?: number | null;
          data_types?: any;
          validation_status?: string;
          validation_errors?: any;
          preprocessing_status?: string;
          processed_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      models: {
        Row: {
          id: string;
          user_id: string;
          project_id: string | null;
          name: string;
          description: string | null;
          architecture: string;
          framework: string;
          base_model: string | null;
          config: any;
          status: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          project_id?: string | null;
          name: string;
          description?: string | null;
          architecture: string;
          framework?: string;
          base_model?: string | null;
          config?: any;
          status?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          project_id?: string | null;
          name?: string;
          description?: string | null;
          architecture?: string;
          framework?: string;
          base_model?: string | null;
          config?: any;
          status?: string;
          created_at?: string;
          updated_at?: string;
        };
      };
      training_runs: {
        Row: {
          id: string;
          user_id: string;
          model_id: string;
          dataset_id: string;
          status: string;
          progress: number;
          current_epoch: number;
          total_epochs: number;
          batch_size: number;
          learning_rate: number;
          loss_function: string;
          optimizer: string;
          metrics: any;
          logs: any;
          model_path: string | null;
          error_message: string | null;
          gpu_used: boolean;
          gpu_memory_used: number | null;
          training_time_seconds: number | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          model_id: string;
          dataset_id: string;
          status?: string;
          progress?: number;
          current_epoch?: number;
          total_epochs?: number;
          batch_size?: number;
          learning_rate?: number;
          loss_function?: string;
          optimizer?: string;
          metrics?: any;
          logs?: any;
          model_path?: string | null;
          error_message?: string | null;
          gpu_used?: boolean;
          gpu_memory_used?: number | null;
          training_time_seconds?: number | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          model_id?: string;
          dataset_id?: string;
          status?: string;
          progress?: number;
          current_epoch?: number;
          total_epochs?: number;
          batch_size?: number;
          learning_rate?: number;
          loss_function?: string;
          optimizer?: string;
          metrics?: any;
          logs?: any;
          model_path?: string | null;
          error_message?: string | null;
          gpu_used?: boolean;
          gpu_memory_used?: number | null;
          training_time_seconds?: number | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      deployments: {
        Row: {
          id: string;
          user_id: string;
          model_id: string;
          training_run_id: string | null;
          name: string;
          description: string | null;
          status: string;
          endpoint_url: string | null;
          api_key: string | null;
          deployment_config: any;
          performance_metrics: any;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          model_id: string;
          training_run_id?: string | null;
          name: string;
          description?: string | null;
          status?: string;
          endpoint_url?: string | null;
          api_key?: string | null;
          deployment_config?: any;
          performance_metrics?: any;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          model_id?: string;
          training_run_id?: string | null;
          name?: string;
          description?: string | null;
          status?: string;
          endpoint_url?: string | null;
          api_key?: string | null;
          deployment_config?: any;
          performance_metrics?: any;
          created_at?: string;
          updated_at?: string;
        };
      };
      metrics: {
        Row: {
          id: string;
          training_run_id: string | null;
          deployment_id: string | null;
          metric_name: string;
          metric_value: number;
          step: number | null;
          epoch: number | null;
          timestamp: string;
        };
        Insert: {
          id?: string;
          training_run_id?: string | null;
          deployment_id?: string | null;
          metric_name: string;
          metric_value: number;
          step?: number | null;
          epoch?: number | null;
          timestamp?: string;
        };
        Update: {
          id?: string;
          training_run_id?: string | null;
          deployment_id?: string | null;
          metric_name?: string;
          metric_value?: number;
          step?: number | null;
          epoch?: number | null;
          timestamp?: string;
        };
      };
      trained_models: {
        Row: {
          id: string;
          user_id: string;
          name: string;
          description: string | null;
          model_type: string;
          dataset_source: string;
          final_loss: number | null;
          final_accuracy: number | null;
          epochs_trained: number;
          model_path: string | null;
          stats: any;
          sandbox_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          name: string;
          description?: string | null;
          model_type: string;
          dataset_source: string;
          final_loss?: number | null;
          final_accuracy?: number | null;
          epochs_trained?: number;
          model_path?: string | null;
          stats?: any;
          sandbox_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          name?: string;
          description?: string | null;
          model_type?: string;
          dataset_source?: string;
          final_loss?: number | null;
          final_accuracy?: number | null;
          epochs_trained?: number;
          model_path?: string | null;
          stats?: any;
          sandbox_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      training_jobs: {
        Row: {
          id: string;
          user_id: string;
          model_id: string | null;
          status: string;
          progress: number;
          current_epoch: number;
          total_epochs: number;
          error_message: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          model_id?: string | null;
          status?: string;
          progress?: number;
          current_epoch?: number;
          total_epochs?: number;
          error_message?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          model_id?: string | null;
          status?: string;
          progress?: number;
          current_epoch?: number;
          total_epochs?: number;
          error_message?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
  };
};
