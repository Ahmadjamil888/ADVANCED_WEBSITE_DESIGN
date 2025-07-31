'use client';

import { cn } from "@/lib/utils";

interface YouTubeVideoProps {
  videoId: string;
  className?: string;
  autoplay?: boolean;
  controls?: boolean;
  mute?: boolean;
  loop?: boolean;
  title?: string;
}

export default function YouTubeVideo({
  videoId,
  className,
  autoplay = true,
  controls = false,
  mute = true,
  loop = true,
  title = "Video",
}: YouTubeVideoProps) {
  // Extract video ID from URL if full URL is provided
  const extractVideoId = (url: string) => {
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);
    return match ? match[1] : url;
  };

  const cleanVideoId = extractVideoId(videoId);

  // Build YouTube embed URL with parameters
  const embedUrl = new URL(`https://www.youtube.com/embed/${cleanVideoId}`);
  
  // Add parameters
  if (autoplay) embedUrl.searchParams.set('autoplay', '1');
  if (!controls) embedUrl.searchParams.set('controls', '0');
  if (mute) embedUrl.searchParams.set('mute', '1');
  if (loop) {
    embedUrl.searchParams.set('loop', '1');
    embedUrl.searchParams.set('playlist', cleanVideoId);
  }
  
  // Additional parameters for clean appearance
  embedUrl.searchParams.set('modestbranding', '1'); // Minimize YouTube branding
  embedUrl.searchParams.set('rel', '0'); // Don't show related videos
  embedUrl.searchParams.set('showinfo', '0'); // Don't show video info
  embedUrl.searchParams.set('iv_load_policy', '3'); // Don't show annotations
  embedUrl.searchParams.set('disablekb', '1'); // Disable keyboard controls
  embedUrl.searchParams.set('fs', '0'); // Disable fullscreen button
  embedUrl.searchParams.set('cc_load_policy', '0'); // Don't show captions
  embedUrl.searchParams.set('playsinline', '1'); // Play inline on mobile
  embedUrl.searchParams.set('start', '0'); // Start from beginning
  embedUrl.searchParams.set('end', ''); // Play full video
  if (typeof window !== 'undefined') {
    embedUrl.searchParams.set('origin', window.location.origin); // Set origin for security
  }

  return (
    <div className={cn("relative w-full aspect-video overflow-hidden rounded-xl bg-black", className)}>
      <iframe
        src={embedUrl.toString()}
        title={title}
        className="absolute inset-0 w-full h-full border-0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowFullScreen={false}
        loading="lazy"
        style={{
          border: 'none',
          outline: 'none',
          background: 'transparent',
        }}
      />
      {/* Subtle overlay to hide YouTube branding */}
      <div 
        className="absolute top-0 right-0 w-20 h-8 bg-gradient-to-l from-black/20 to-transparent pointer-events-none z-10"
      />
      {/* Bottom overlay to hide YouTube controls area */}
      <div 
        className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-black/10 to-transparent pointer-events-none z-10"
      />
    </div>
  );
}