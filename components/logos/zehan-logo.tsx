import { cn } from "@/lib/utils";

interface ZehanLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export default function ZehanLogo({ className, size = "md" }: ZehanLogoProps) {
  const sizeClasses = {
    sm: "w-6 h-6",
    md: "w-8 h-8", 
    lg: "w-12 h-12"
  };

  return (
    <div className={cn("relative flex items-center justify-center", sizeClasses[size], className)}>
      <svg
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        {/* Fancy Z made with three lines */}
        <g className="text-white">
          {/* Top horizontal line */}
          <rect
            x="6"
            y="8"
            width="20"
            height="2.5"
            rx="1.25"
            fill="currentColor"
            className="drop-shadow-sm"
          />
          
          {/* Middle diagonal line */}
          <rect
            x="8"
            y="14.75"
            width="16"
            height="2.5"
            rx="1.25"
            fill="currentColor"
            transform="rotate(-26.57 16 16)"
            className="drop-shadow-sm"
          />
          
          {/* Bottom horizontal line */}
          <rect
            x="6"
            y="21.5"
            width="20"
            height="2.5"
            rx="1.25"
            fill="currentColor"
            className="drop-shadow-sm"
          />
        </g>
        
        {/* Subtle glow effect */}
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="1" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
      </svg>
    </div>
  );
}