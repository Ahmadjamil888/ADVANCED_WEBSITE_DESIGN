'use client';

import Image from 'next/image';
import { useState } from 'react';
import { cn } from '@/lib/utils';

interface LazyImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  priority?: boolean;
  fill?: boolean;
}

export default function LazyImage({
  src,
  alt,
  width,
  height,
  className,
  priority = false,
  fill = false,
}: LazyImageProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  return (
    <div className={cn('relative overflow-hidden', className)}>
      {!hasError ? (
        <Image
          src={src}
          alt={alt}
          width={width}
          height={height}
          fill={fill}
          priority={priority}
          className={cn(
            'transition-opacity duration-300',
            isLoading ? 'opacity-0' : 'opacity-100'
          )}
          onLoad={() => setIsLoading(false)}
          onError={() => {
            setIsLoading(false);
            setHasError(true);
          }}
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
        />
      ) : (
        <div className="flex items-center justify-center bg-muted text-muted-foreground p-4">
          <span className="text-sm">Image not available</span>
        </div>
      )}
      
      {isLoading && (
        <div className="absolute inset-0 bg-muted animate-pulse" />
      )}
    </div>
  );
}