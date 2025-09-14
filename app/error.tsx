'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Section } from '@/components/ui/section';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import Link from 'next/link';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Application error:', error);
  }, [error]);

  return (
    <Section className="min-h-screen flex items-center justify-center">
      <div className="max-w-md mx-auto text-center">
        <div className="size-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-6">
          <AlertTriangle className="size-8 text-red-600 dark:text-red-400" />
        </div>
        
        <h1 className="text-2xl font-bold mb-4">Something went wrong!</h1>
        
        <p className="text-muted-foreground mb-8">
          We apologize for the inconvenience. An unexpected error has occurred.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button onClick={reset} className="flex items-center gap-2">
            <RefreshCw className="size-4" />
            Try Again
          </Button>
          
          <Button variant="outline" asChild>
            <Link href="/" className="flex items-center gap-2">
              <Home className="size-4" />
              Go Home
            </Link>
          </Button>
        </div>
        
        <div className="mt-8 p-4 bg-muted/30 rounded-lg">
          <p className="text-sm text-muted-foreground">
            If this problem persists, please contact us at{' '}
              <a 
                href="/contact" 
                className="text-primary hover:underline"
              >
                Contact Us
              </a>
          </p>
        </div>
      </div>
    </Section>
  );
}