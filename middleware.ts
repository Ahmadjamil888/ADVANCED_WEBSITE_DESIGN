import { NextResponse, NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const url = request.nextUrl.clone();
  const host = request.headers.get('host') || '';

  // Handle wildcard subdomains like fallback-<id>.zehanxtech.com
  if (host.endsWith('zehanxtech.com')) {
    const parts = host.split('.');
    // e.g., [ 'fallback-abc12345', 'zehanxtech', 'com' ]
    if (parts.length >= 3) {
      const sub = parts[0];
      if (sub && sub !== 'www') {
        // If it's a fallback-* subdomain, rewrite to internal informative page
        const shortId = sub.startsWith('fallback-') ? sub.replace('fallback-', '') : sub;
        url.pathname = `/e2b-fallback/${shortId}`;
        return NextResponse.rewrite(url);
      }
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
