"use client";

import { Share2, Twitter, Linkedin, Facebook } from "lucide-react";
import { Button } from "./button";

interface SocialShareProps {
  url?: string;
  title?: string;
  description?: string;
  className?: string;
}

export default function SocialShare({ 
  url = typeof window !== 'undefined' ? window.location.href : '',
  title = "Zehan X Technologies - AI & Web Development Experts",
  description = "Expert AI & web development company. Transform your business with cutting-edge AI technology.",
  className = ""
}: SocialShareProps) {
  const encodedUrl = encodeURIComponent(url);
  const encodedTitle = encodeURIComponent(title);
  const shareLinks = {
    twitter: `https://twitter.com/intent/tweet?url=${encodedUrl}&text=${encodedTitle}`,
    linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`,
    facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`,
  };

  const handleNativeShare = async () => {
    if (typeof navigator !== 'undefined' && navigator.share) {
      try {
        await navigator.share({
          title,
          text: description,
          url,
        });
      } catch (error) {
        console.log('Error sharing:', error);
      }
    }
  };

  const hasNativeShare = typeof navigator !== 'undefined' && 'share' in navigator;

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="text-sm text-muted-foreground mr-2">Share:</span>
      
      {hasNativeShare && (
        <Button
          variant="outline"
          size="sm"
          onClick={handleNativeShare}
          className="flex items-center gap-1"
        >
          <Share2 className="size-4" />
          Share
        </Button>
      )}
      
      <Button
        variant="outline"
        size="sm"
        asChild
      >
        <a
          href={shareLinks.twitter}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1"
        >
          <Twitter className="size-4" />
          Twitter
        </a>
      </Button>
      
      <Button
        variant="outline"
        size="sm"
        asChild
      >
        <a
          href={shareLinks.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1"
        >
          <Linkedin className="size-4" />
          LinkedIn
        </a>
      </Button>
      
      <Button
        variant="outline"
        size="sm"
        asChild
      >
        <a
          href={shareLinks.facebook}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1"
        >
          <Facebook className="size-4" />
          Facebook
        </a>
      </Button>
    </div>
  );
}