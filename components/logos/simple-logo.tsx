"use client";

import { cn } from "@/lib/utils";
import { useState } from "react";

interface SimpleLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export default function SimpleLogo({ className, size = "md" }: SimpleLogoProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  
  const sizeClasses = {
    sm: "w-10 h-10",
    md: "w-12 h-12", 
    lg: "w-16 h-16"
  };

  // Fallback logo if image fails to load
  const FallbackLogo = () => (
    <div className="w-full h-full bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center">
      <span className="text-white font-bold text-lg">Z</span>
    </div>
  );

  return (
    <div className={cn("relative flex items-center justify-center overflow-hidden rounded-lg", sizeClasses[size], className)}>
      {imageError ? (
        <FallbackLogo />
      ) : (
        <>
          <img
            src="/logo.jpg"
            alt="Company Logo"
            className={cn(
              "w-full h-full object-cover rounded-lg transition-opacity duration-300",
              imageLoaded ? "opacity-100" : "opacity-0"
            )}
            style={{ objectFit: 'cover' }}
            onError={() => {
              console.log('Logo failed to load, showing fallback');
              setImageError(true);
            }}
            onLoad={() => {
              console.log('Logo loaded successfully');
              setImageLoaded(true);
            }}
          />
          {!imageLoaded && !imageError && (
            <div className="absolute inset-0 bg-gray-200 animate-pulse rounded-lg" />
          )}
        </>
      )}
    </div>
  );
}