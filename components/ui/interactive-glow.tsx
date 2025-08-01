"use client";

import { useEffect, useRef, useState } from "react";

interface InteractiveGlowProps {
  className?: string;
}

export default function InteractiveGlow({ className }: InteractiveGlowProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setMousePosition({ x, y });
      }
    };

    const handleMouseEnter = () => setIsHovering(true);
    const handleMouseLeave = () => setIsHovering(false);

    const container = containerRef.current;
    if (container) {
      container.addEventListener("mousemove", handleMouseMove);
      container.addEventListener("mouseenter", handleMouseEnter);
      container.addEventListener("mouseleave", handleMouseLeave);

      return () => {
        container.removeEventListener("mousemove", handleMouseMove);
        container.removeEventListener("mouseenter", handleMouseEnter);
        container.removeEventListener("mouseleave", handleMouseLeave);
      };
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}
      style={{ pointerEvents: 'auto' }}
    >
      {/* Base ambient glow */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/8 via-transparent to-primary/5 pointer-events-none"></div>
      
      {/* Floating orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/15 rounded-full blur-3xl animate-pulse pointer-events-none"></div>
      <div className="absolute top-3/4 right-1/4 w-80 h-80 bg-primary/12 rounded-full blur-2xl animate-pulse delay-1000 pointer-events-none"></div>
      <div className="absolute top-1/2 left-3/4 w-64 h-64 bg-primary/18 rounded-full blur-2xl animate-pulse delay-2000 pointer-events-none"></div>
      
      {/* Mouse-following glow */}
      <div
        className={`absolute w-[600px] h-[600px] rounded-full transition-all duration-300 ease-out pointer-events-none ${
          isHovering ? "opacity-40" : "opacity-20"
        }`}
        style={{
          left: `${mousePosition.x}%`,
          top: `${mousePosition.y}%`,
          transform: "translate(-50%, -50%)",
          background: `radial-gradient(circle, oklch(45% 0.15 240 / 0.4) 0%, oklch(45% 0.15 240 / 0.2) 30%, transparent 70%)`,
          filter: "blur(40px)",
        }}
      ></div>
      
      {/* Secondary mouse glow */}
      <div
        className={`absolute w-[400px] h-[400px] rounded-full transition-all duration-500 ease-out pointer-events-none ${
          isHovering ? "opacity-50" : "opacity-25"
        }`}
        style={{
          left: `${mousePosition.x}%`,
          top: `${mousePosition.y}%`,
          transform: "translate(-50%, -50%)",
          background: `radial-gradient(circle, oklch(50% 0.18 260 / 0.5) 0%, oklch(50% 0.18 260 / 0.3) 40%, transparent 80%)`,
          filter: "blur(20px)",
        }}
      ></div>
      
      {/* Cursor spotlight */}
      <div
        className={`absolute w-[200px] h-[200px] rounded-full transition-all duration-200 ease-out pointer-events-none ${
          isHovering ? "opacity-60" : "opacity-0"
        }`}
        style={{
          left: `${mousePosition.x}%`,
          top: `${mousePosition.y}%`,
          transform: "translate(-50%, -50%)",
          background: `radial-gradient(circle, oklch(70% 0.12 240 / 0.7) 0%, oklch(70% 0.12 240 / 0.4) 50%, transparent 100%)`,
          filter: "blur(10px)",
        }}
      ></div>
    </div>
  );
}