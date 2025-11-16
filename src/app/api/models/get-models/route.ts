import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const maxDuration = 60;

export async function GET(request: NextRequest) {
  try {
    const userId = request.nextUrl.searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        { error: 'userId is required' },
        { status: 400 }
      );
    }

    // Initialize Supabase client
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json(
        { error: 'Supabase configuration missing' },
        { status: 500 }
      );
    }

    const supabase = createClient(supabaseUrl, supabaseKey);

    // Fetch user's models
    const { data, error } = await supabase
      .from('ai_models')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('[get-models] Supabase error:', error);
      return NextResponse.json(
        { error: `Failed to fetch models: ${error.message}` },
        { status: 500 }
      );
    }

    console.log('[get-models] Fetched models for user:', userId, 'Count:', data?.length || 0);

    return NextResponse.json({
      success: true,
      models: data || [],
      count: data?.length || 0,
    });
  } catch (error) {
    console.error('[get-models] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to fetch models',
      },
      { status: 500 }
    );
  }
}
