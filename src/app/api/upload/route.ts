import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseOrThrow } from '@/lib/supabase';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const type = formData.get('type') as string;

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const supabase = getSupabaseOrThrow();
    const user = (await supabase.auth.getUser()).data.user;
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Save file to uploads directory
    const uploadsDir = join(process.cwd(), 'uploads', user.id);
    await mkdir(uploadsDir, { recursive: true });

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filename = `${Date.now()}-${file.name}`;
    const filepath = join(uploadsDir, filename);

    await writeFile(filepath, buffer);

    // Save to database
    const table = type === 'dataset' ? 'user_datasets' : 'user_models';
    const { data, error } = await (supabase.from(table).insert as any)({
      user_id: user.id,
      name: file.name,
      file_path: filepath,
      file_size: file.size,
      file_type: file.type,
      source_type: 'upload',
    }).select().single();

    if (error) throw error;

    return NextResponse.json({ success: true, path: filepath, id: data.id });
  } catch (error: any) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

