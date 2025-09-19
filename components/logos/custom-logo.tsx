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

  return (
    <div className={cn("relative flex items-center justify-center", sizeClasses[size], className)}>
      <Image
        src="/logo.jpg"
        alt="Logo"
        width={48}
        height={48}
        className="w-full h-full object-contain rounded-md"
        priority
      />
    </div>
  );
}