import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function POST(request: NextRequest) {
  try {
    const { userId, hfToken } = await request.json()

    if (!userId || !hfToken) {
      return NextResponse.json({ error: 'User ID and HF token are required' }, { status: 400 })
    }

    if (!supabase) {
      return NextResponse.json({ error: 'Database not available' }, { status: 500 })
    }

    // Save or update HF token in user_integrations table
    const { data, error } = await (supabase
      .from('user_integrations')
      .upsert as any)({
        user_id: userId,
        service_name: 'huggingface',
        encrypted_api_key: hfToken, // In production, encrypt this
        is_active: true,
        last_used_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }, {
        onConflict: 'user_id,service_name'
      })
      .select()

    if (error) {
      console.error('Error saving HF token:', error)
      return NextResponse.json({ error: 'Failed to save token' }, { status: 500 })
    }

    return NextResponse.json({ 
      success: true,
      message: 'Hugging Face token saved successfully'
    })

  } catch (error: any) {
    console.error('HF token save error:', error)
    
    return NextResponse.json(
      { error: `Failed to save token: ${error.message}` },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 })
    }

    if (!supabase) {
      return NextResponse.json({ error: 'Database not available' }, { status: 500 })
    }

    // Get HF token from user_integrations table
    const { data, error } = await supabase
      .from('user_integrations')
      .select('encrypted_api_key, is_active')
      .eq('user_id', userId)
      .eq('service_name', 'huggingface')
      .single()

    if (error && error.code !== 'PGRST116') { // PGRST116 is "not found"
      console.error('Error getting HF token:', error)
      return NextResponse.json({ error: 'Failed to get token' }, { status: 500 })
    }

    return NextResponse.json({ 
      hasToken: !!(data as any)?.encrypted_api_key,
      isActive: (data as any)?.is_active || false,
      token: (data as any)?.encrypted_api_key || null
    })

  } catch (error: any) {
    console.error('HF token get error:', error)
    
    return NextResponse.json(
      { error: `Failed to get token: ${error.message}` },
      { status: 500 }
    )
  }
}