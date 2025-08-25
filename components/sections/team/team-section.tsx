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
        'py-20 bg-gradient-to-br from-black via-gray-900 to-black relative overflow-hidden',
        className
      )}
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
      </div>
      
      <div ref={sectionRef} className="max-w-6xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="team-title text-4xl md:text-6xl font-bold text-white mb-6">
            Meet Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#A607F2] to-purple-400">Team</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Led by visionary leadership and powered by innovative minds
          </p>
        </div>

        <div className="flex justify-center">
          <div className="team-card group relative">
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-[#A607F2]/30 hover:border-[#A607F2]/60 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-[#A607F2]/20 max-w-md">
              {/* Profile Image Placeholder */}
              <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-[#A607F2] to-purple-600 flex items-center justify-center text-4xl font-bold text-white">
                AJ
              </div>
              
              <div className="text-center">
                <h3 className="text-2xl font-bold text-white mb-2">Ahmad Jamil</h3>
                <p className="text-[#A607F2] font-semibold mb-4">Founder & CEO</p>
                <p className="text-gray-300 leading-relaxed">
                  Visionary leader driving AI innovation and business transformation. 
                  With expertise in cutting-edge technology and strategic business development, 
                  Ahmad leads our mission to revolutionize how businesses leverage artificial intelligence.
                </p>
              </div>
              
              {/* Hover Effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-[#A607F2]/10 to-purple-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}