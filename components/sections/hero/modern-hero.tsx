'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Typewriter from 'typewriter-effect';
import { useInView } from 'react-intersection-observer';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Section } from '../../ui/section';
import Spline3D from '../../ui/spline-3d';
import { cn } from '@/lib/utils';

// Register GSAP plugins
if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface ModernHeroProps {
  className?: string;
}

export default function ModernHero({ className }: ModernHeroProps) {
  const heroRef = useRef<HTMLDivElement>(null);
  const vantaRef = useRef<HTMLDivElement>(null);
  const [vantaEffect, setVantaEffect] = useState<{ destroy: () => void } | null>(null);
  const { ref: inViewRef, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  useEffect(() => {
    // Initialize Vanta.js background
    const initVanta = async () => {
      if (typeof window !== 'undefined' && vantaRef.current) {
        const { default: VANTA } = await import('vanta/dist/vanta.net.min.js');
        const THREE = await import('three');
        
        const effect = VANTA({
          el: vantaRef.current,
          THREE,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xa607f2,
          backgroundColor: 0x0a0a0a,
          points: 10.00,
          maxDistance: 20.00,
          spacing: 15.00
        });
        
        setVantaEffect(effect);
      }
    };

    initVanta();

    return () => {
      if (vantaEffect) {
        vantaEffect.destroy();
      }
    };
  }, [vantaEffect]);

  useEffect(() => {
    if (inView && heroRef.current) {
      const tl = gsap.timeline();
      
      // Animate elements on scroll
      tl.fromTo('.hero-badge', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
      )
      .fromTo('.hero-title', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' },
        '-=0.4'
      )
      .fromTo('.hero-subtitle', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' },
        '-=0.4'
      )
      .fromTo('.hero-buttons', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' },
        '-=0.4'
      );

      // Floating animation for elements
      gsap.to('.floating-element', {
        y: -20,
        duration: 2,
        ease: 'power2.inOut',
        yoyo: true,
        repeat: -1,
        stagger: 0.2
      });
    }
  }, [inView]);

  const setRefs = useCallback((node: HTMLDivElement) => {
    heroRef.current = node;
    inViewRef(node);
  }, [inViewRef]);

  return (
    <Section
      ref={setRefs}
      className={cn(
        'relative min-h-screen overflow-hidden flex items-center justify-center',
        className
      )}
    >
      {/* Vanta.js Background */}
      <div 
        ref={vantaRef}
        className="absolute inset-0 z-0"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-black/50 via-transparent to-purple-900/30 z-10" />
      
      {/* Content */}
      <div ref={heroRef} className="relative z-20 max-w-6xl mx-auto px-4 text-center">
        <Badge 
          variant="outline" 
          className="hero-badge mb-8 border-[#A607F2] text-[#A607F2] bg-black/20 backdrop-blur-sm floating-element"
        >
          <span className="font-semibold">AI-Powered Business Solutions</span>
        </Badge>
        
        <h1 className="hero-title text-4xl md:text-6xl lg:text-8xl font-bold mb-8 leading-tight">
          <span className="text-white">
            <Typewriter
              options={{
                strings: ['Automate Your Business with AI'],
                autoStart: true,
                loop: false,
                delay: 100,
                cursor: '|',
                wrapperClassName: 'font-bold',
              }}
            />
          </span>
        </h1>
        
        <p className="hero-subtitle text-xl md:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed floating-element">
          Transform your business with cutting-edge AI solutions, custom web development, 
          and intelligent automation systems that drive real results.
        </p>
        
        <div className="hero-buttons flex flex-col sm:flex-row gap-6 justify-center items-center floating-element">
          <Button
            size="lg"
            className="bg-gradient-to-r from-[#A607F2] to-purple-600 hover:from-[#8A05D1] hover:to-purple-700 text-white px-8 py-4 text-lg font-semibold rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
            asChild
          >
            <a href="/contact">Get In Touch</a>
          </Button>
          
          <Button
            variant="outline"
            size="lg"
            className="border-[#A607F2] text-[#A607F2] hover:bg-[#A607F2] hover:text-white px-8 py-4 text-lg font-semibold rounded-full backdrop-blur-sm bg-black/20 transition-all duration-300 transform hover:scale-105"
            asChild
          >
            <a href="/portfolio">View Our Work</a>
          </Button>
        </div>
      </div>
      
      {/* 3D Spline Elements */}
      <Spline3D className="absolute inset-0 z-10" />
      
      {/* Floating Elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-gradient-to-br from-[#A607F2] to-purple-600 rounded-full opacity-20 floating-element" />
      <div className="absolute bottom-20 right-10 w-16 h-16 bg-gradient-to-br from-[#A607F2] to-purple-600 rounded-full opacity-20 floating-element" />
      <div className="absolute top-1/2 left-20 w-12 h-12 bg-gradient-to-br from-[#A607F2] to-purple-600 rounded-full opacity-20 floating-element" />
    </Section>
  );
}