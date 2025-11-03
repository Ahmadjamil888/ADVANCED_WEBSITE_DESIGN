import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { createClient } from '@supabase/supabase-js';

// Create a service role client for admin operations
const supabaseAdmin = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Missing or invalid authorization header' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    
    // Get user from token
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token);
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    // Fetch user's API keys - using flexible column selection
    const { data: apiKeys, error } = await supabaseAdmin
      .from('api_keys')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching API keys:', error);
      return NextResponse.json({ error: 'Failed to fetch API keys' }, { status: 500 });
    }

    // Transform the data to ensure consistent format
    const transformedKeys = (apiKeys || []).map(key => ({
      id: key.id,
      name: key.name || 'Unnamed Key',
      description: key.description || '',
      key_preview: key.key_preview || key.api_key?.substring(0, 12) + '...' || 'zx_***...',
      is_active: key.is_active !== false,
      is_revoked: key.is_revoked || false,
      max_daily_requests: key.max_daily_requests || 1000,
      max_monthly_requests: key.max_monthly_requests || 10000,
      total_usage: key.total_usage || 0,
      daily_usage: key.daily_usage || 0,
      monthly_usage: key.monthly_usage || 0,
      last_used_at: key.last_used_at || null,
      created_at: key.created_at
    }));

    return NextResponse.json({ apiKeys: transformedKeys });

  } catch (error) {
    console.error('API Keys GET error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Missing or invalid authorization header' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    
    // Get user from token
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token);
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    const body = await request.json();
    const { name, description } = body;

    if (!name?.trim()) {
      return NextResponse.json({ error: 'API key name is required' }, { status: 400 });
    }

    // Generate API key
    const generateApiKey = () => {
      const prefix = 'zx_';
      const randomBytes = new Uint8Array(32);
      crypto.getRandomValues(randomBytes);
      const randomPart = Array.from(randomBytes)
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');
      return prefix + randomPart;
    };

    const apiKey = generateApiKey();
    const keyPreview = apiKey.substring(0, 12) + '...';
    
    // Hash the key for storage (using btoa for simplicity, use proper hashing in production)
    const keyHash = btoa(apiKey);

    // Try to insert with flexible column names
    const insertData: any = {
      user_id: user.id,
      name: name.trim(),
      api_key: apiKey, // Some tables might store the full key
      key_preview: keyPreview,
      is_active: true,
      created_at: new Date().toISOString()
    };

    // Add optional fields if they exist
    if (description?.trim()) {
      insertData.description = description.trim();
    }

    // Try common column names
    try {
      insertData.key_hash = keyHash;
    } catch (e) {
      // Column might not exist
    }

    try {
      insertData.max_daily_requests = 1000;
      insertData.max_monthly_requests = 10000;
      insertData.max_tokens_per_request = 2048;
    } catch (e) {
      // Columns might not exist
    }

    // Insert API key into database
    const { data: newApiKey, error } = await supabaseAdmin
      .from('api_keys')
      .insert(insertData)
      .select('*')
      .single();

    if (error) {
      console.error('Error creating API key:', error);
      return NextResponse.json({ error: `Failed to create API key: ${error.message}` }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      apiKey: {
        id: newApiKey.id,
        name: newApiKey.name,
        description: newApiKey.description || '',
        key_preview: newApiKey.key_preview,
        is_active: newApiKey.is_active,
        max_daily_requests: newApiKey.max_daily_requests || 1000,
        max_monthly_requests: newApiKey.max_monthly_requests || 10000,
        created_at: newApiKey.created_at,
        key: apiKey // Return the actual key only once
      }
    });

  } catch (error) {
    console.error('API Keys POST error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Missing or invalid authorization header' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    
    // Get user from token
    const { data: { user }, error: authError } = await supabaseAdmin.auth.getUser(token);
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const keyId = searchParams.get('id');

    if (!keyId) {
      return NextResponse.json({ error: 'API key ID is required' }, { status: 400 });
    }

    // Try to revoke the API key (soft delete) or hard delete if no revoke column
    const updateData: any = {
      is_active: false,
      updated_at: new Date().toISOString()
    };

    // Try to set is_revoked if column exists
    try {
      updateData.is_revoked = true;
    } catch (e) {
      // Column might not exist, that's okay
    }

    const { error } = await supabaseAdmin
      .from('api_keys')
      .update(updateData)
      .eq('id', keyId)
      .eq('user_id', user.id);

    if (error) {
      // If update fails, try delete
      const { error: deleteError } = await supabaseAdmin
        .from('api_keys')
        .delete()
        .eq('id', keyId)
        .eq('user_id', user.id);

      if (deleteError) {
        console.error('Error deleting API key:', deleteError);
        return NextResponse.json({ error: 'Failed to revoke API key' }, { status: 500 });
      }
    }

    return NextResponse.json({ success: true });

  } catch (error) {
    console.error('API Keys DELETE error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}