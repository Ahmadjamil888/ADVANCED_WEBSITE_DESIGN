import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();

    // Kaggle API requires authentication, but we can use their public search
    // For production, you'd need KAGGLE_USERNAME and KAGGLE_KEY env vars
    const kaggleUsername = process.env.KAGGLE_USERNAME;
    const kaggleKey = process.env.KAGGLE_KEY;

    if (!kaggleUsername || !kaggleKey) {
      // Fallback: return a search URL for manual selection
      const searchQuery = encodeURIComponent(prompt);
      return NextResponse.json({
        success: true,
        name: 'kaggle_search',
        url: `https://www.kaggle.com/datasets?search=${searchQuery}`,
        message: 'Kaggle API credentials not configured. Please select dataset manually.',
        requiresManualSelection: true,
      });
    }

    // If credentials are available, use Kaggle API
    const searchQuery = encodeURIComponent(prompt);
    const auth = Buffer.from(`${kaggleUsername}:${kaggleKey}`).toString('base64');

    const response = await fetch(
      `https://www.kaggle.com/api/v1/datasets/search?search=${searchQuery}&pageSize=5`,
      {
        headers: {
          'Authorization': `Basic ${auth}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Kaggle API error: ${response.status}`);
    }

    const data = await response.json();

    if (data && data.length > 0) {
      const bestMatch = data[0];
      return NextResponse.json({
        success: true,
        name: bestMatch.ref,
        repo: bestMatch.ref,
        downloads: bestMatch.downloadCount || 0,
        description: bestMatch.title || '',
        url: `https://www.kaggle.com/datasets/${bestMatch.ref}`,
      });
    }

    return NextResponse.json({
      success: false,
      message: 'No matching datasets found',
    });
  } catch (error: any) {
    console.error('Kaggle scraping error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

