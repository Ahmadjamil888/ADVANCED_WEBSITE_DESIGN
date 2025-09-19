"use client";

import { cn } from "@/lib/utils";

interface SimpleLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export default function SimpleLogo({ className, size = "md" }: SimpleLogoProps) {
  const sizeClasses = {
    sm: "w-8 h-8",
    md: "w-10 h-10", 
    lg: "w-12 h-12"
  };

  return (
    <div className={cn("relative flex items-center justify-center overflow-hidden rounded-lg", sizeClasses[size], className)}>
      <img
        src="/logo.jpg"
        alt="Company Logo"
        className="w-full h-full object-cover"
        style={{ objectFit: 'cover' }}
      />
    </div>
  );
}