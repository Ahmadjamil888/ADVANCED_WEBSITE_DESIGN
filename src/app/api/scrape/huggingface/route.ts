import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { prompt, type } = await req.json(); // type: 'dataset' | 'model'

    // Use HuggingFace Hub API (no auth required for public repos)
    const searchQuery = encodeURIComponent(prompt);
    const apiUrl = type === 'dataset'
      ? `https://huggingface.co/api/datasets?search=${searchQuery}&sort=downloads&direction=-1&limit=5`
      : `https://huggingface.co/api/models?search=${searchQuery}&sort=downloads&direction=-1&limit=5`;

    const response = await fetch(apiUrl, {
      headers: {
        'User-Agent': 'zehanxtech-ai-platform/1.0',
      },
    });

    if (!response.ok) {
      throw new Error(`HF API error: ${response.status}`);
    }

    const data = await response.json();

    // Return best match
    if (data && data.length > 0) {
      const bestMatch = data[0];
      return NextResponse.json({
        success: true,
        name: bestMatch.id,
        repo: bestMatch.id,
        downloads: bestMatch.downloads || 0,
        description: bestMatch.description || '',
        url: `https://huggingface.co/${bestMatch.id}`,
      });
    }

    return NextResponse.json({
      success: false,
      message: 'No matching resources found',
    });
  } catch (error: any) {
    console.error('HF scraping error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

