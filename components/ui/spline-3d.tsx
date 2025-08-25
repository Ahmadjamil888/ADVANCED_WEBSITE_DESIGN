'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

interface Spline3DProps {
  className?: string;
}

export default function Spline3D({ className }: Spline3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      // Create animated 3D-like elements as placeholder
      const elements = containerRef.current.querySelectorAll('.spline-element');
      
      gsap.to(elements, {
        rotationY: 360,
        duration: 10,
        ease: 'none',
        repeat: -1,
        stagger: 0.5
      });

      gsap.to(elements, {
        y: -20,
        duration: 3,
        ease: 'power2.inOut',
        yoyo: true,
        repeat: -1,
        stagger: 0.3
      });
    }
  }, []);

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* 3D-like animated elements */}
      <div className="spline-element absolute top-10 left-10 w-16 h-16 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg opacity-70 transform rotate-12" />
      <div className="spline-element absolute top-20 right-20 w-12 h-12 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-full opacity-60 transform -rotate-12" />
      <div className="spline-element absolute bottom-20 left-20 w-20 h-20 bg-gradient-to-br from-blue-600 to-cyan-400 rounded-xl opacity-50 transform rotate-45" />
      <div className="spline-element absolute bottom-10 right-10 w-14 h-14 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg opacity-80 transform -rotate-45" />
      
      {/* Central floating element */}
      <div className="spline-element absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-24 h-24 bg-gradient-to-br from-blue-600 via-cyan-500 to-cyan-600 rounded-2xl opacity-40 shadow-2xl shadow-blue-600/30" />
    </div>
  );
}