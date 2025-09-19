"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";

interface CustomLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export default function CustomLogo({ className, size = "md" }: CustomLogoProps) {
  const sizeClasses = {
    sm: "w-8 h-8",
    md: "w-10 h-10", 
    lg: "w-12 h-12"
  };

  const sizeValues = {
    sm: 32,
    md: 40,
    lg: 48
  };

  return (
    <div className={cn("relative flex items-center justify-center overflow-hidden", sizeClasses[size], className)}>
      <Image
        src="/logo.jpg"
        alt="Company Logo"
        width={sizeValues[size]}
        height={sizeValues[size]}
        className="w-full h-full object-cover rounded-lg"
        priority
        style={{ objectFit: 'cover' }}
      />
    </div>
  );
}