'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useInView } from 'react-intersection-observer';
import { Section } from '../../ui/section';
import { cn } from '@/lib/utils';
import { Button } from '../../ui/button';
import { ArrowRight, Users, Target, Zap } from 'lucide-react';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface CorporateSectionProps {
  className?: string;
  title: string;
  subtitle: string;
  description: string;
  imageUrl: string;
  imageAlt: string;
  buttonText?: string;
  buttonHref?: string;
  features?: string[];
  reverse?: boolean;
}

export default function CorporateSection({ 
  className,
  title,
  subtitle,
  description,
  imageUrl,
  imageAlt,
  buttonText = "Learn More",
  buttonHref = "/contact",
  features = [],
  reverse = false
}: CorporateSectionProps) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const { ref: inViewRef, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  useEffect(() => {
    if (inView && sectionRef.current) {
      const tl = gsap.timeline();
      
      tl.fromTo('.corporate-content', 
        { opacity: 0, x: reverse ? 50 : -50 },
        { opacity: 1, x: 0, duration: 0.8, ease: 'power2.out' }
      )
      .fromTo('.corporate-image', 
        { opacity: 0, x: reverse ? -50 : 50 },
        { opacity: 1, x: 0, duration: 0.8, ease: 'power2.out' },
        '-=0.4'
      );
    }
  }, [inView, reverse]);

  return (
    <Section
      ref={inViewRef}
      className={cn(
        'py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden',
        className
      )}
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary to-transparent" />
      </div>
      
      <div ref={sectionRef} className="max-w-7xl mx-auto px-4 relative z-10">
        <div className={cn(
          "grid lg:grid-cols-2 gap-12 items-center",
          reverse && "lg:grid-flow-col-dense"
        )}>
          {/* Content Side */}
          <div className={cn(
            "corporate-content space-y-6",
            reverse && "lg:col-start-2"
          )}>
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-primary uppercase tracking-wider">
                {subtitle}
              </h3>
              <h2 className="text-4xl md:text-5xl font-bold text-white leading-tight">
                {title}
              </h2>
              <p className="text-xl text-gray-300 leading-relaxed">
                {description}
              </p>
            </div>

            {features.length > 0 && (
              <div className="space-y-3">
                {features.map((feature, index) => (
                  <div key={index} className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-primary rounded-full flex-shrink-0" />
                    <span className="text-gray-300">{feature}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Button 
                asChild
                className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-3 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <a href={buttonHref} className="inline-flex items-center gap-2">
                  {buttonText}
                  <ArrowRight className="w-4 h-4" />
                </a>
              </Button>
              <Button 
                variant="outline"
                asChild
                className="border-white/30 text-white hover:bg-white/10 px-8 py-3 rounded-full font-semibold transition-all duration-300"
              >
                <a href="/contact">
                  Get Started
                </a>
              </Button>
            </div>
          </div>

          {/* Image Side */}
          <div className={cn(
            "corporate-image relative",
            reverse && "lg:col-start-1"
          )}>
            <div className="relative rounded-2xl overflow-hidden shadow-2xl">
              <img 
                src={imageUrl}
                alt={imageAlt}
                className="w-full h-[400px] lg:h-[500px] object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
            </div>
            
            {/* Decorative Elements */}
            <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-primary/20 to-primary/10 rounded-full blur-xl" />
            <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-gradient-to-br from-cyan-500/20 to-blue-500/10 rounded-full blur-xl" />
          </div>
        </div>
      </div>
    </Section>
  );
}