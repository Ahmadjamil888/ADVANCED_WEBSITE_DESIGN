'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useInView } from 'react-intersection-observer';
import { Section } from '../../ui/section';
import { cn } from '@/lib/utils';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface TeamSectionProps {
  className?: string;
}

export default function TeamSection({ className }: TeamSectionProps) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const { ref: inViewRef, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  useEffect(() => {
    if (inView && sectionRef.current) {
      const tl = gsap.timeline();
      
      tl.fromTo('.team-title', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
      )
      .fromTo('.team-card', 
        { opacity: 0, y: 50, scale: 0.9 },
        { opacity: 1, y: 0, scale: 1, duration: 0.8, ease: 'power2.out', stagger: 0.2 },
        '-=0.4'
      );
    }
  }, [inView]);

  return (
    <Section
      ref={inViewRef}
      className={cn(
        'py-20 gradient-bg-professional relative overflow-hidden',
        className
      )}
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary to-transparent" />
      </div>
      
      <div ref={sectionRef} className="max-w-6xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="team-title text-4xl md:text-6xl font-bold text-foreground mb-6 heading-professional">
            Meet Our <span className="text-gradient-primary">Team</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto subheading-professional">
            Led by visionary leadership and powered by innovative minds
          </p>
        </div>

        <div className="flex justify-center">
          <div className="team-card group relative">
            <div className="professional-card rounded-2xl p-8 border border-primary/30 hover:border-primary/60 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-primary/20 max-w-md hover-lift-professional">
              {/* Profile Image Placeholder */}
              <div className="w-32 h-32 mx-auto mb-6 rounded-full btn-gradient-primary flex items-center justify-center text-4xl font-bold text-white">
                AJ
              </div>
              
              <div className="text-center">
                <h3 className="text-2xl font-bold text-foreground mb-2 heading-professional">Ahmad Jamil</h3>
                <p className="text-primary font-semibold mb-4">Founder & CEO</p>
                <p className="text-muted-foreground leading-relaxed">
                  Visionary leader driving AI innovation and business transformation. 
                  With expertise in cutting-edge technology and strategic business development, 
                  Ahmad leads our mission to revolutionize how businesses leverage artificial intelligence.
                </p>
              </div>
              
              {/* Hover Effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary/10 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}