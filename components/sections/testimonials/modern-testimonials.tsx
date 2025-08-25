'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useInView } from 'react-intersection-observer';
import { Star } from 'lucide-react';
import { Section } from '../../ui/section';
import { cn } from '@/lib/utils';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface TestimonialsProps {
  className?: string;
}

const testimonials = [
  {
    name: 'Sarah Johnson',
    role: 'CEO, TechStart Inc.',
    content: 'The AI solutions provided by this team transformed our business operations. We saw a 40% increase in efficiency within the first month.',
    rating: 5,
  },
  {
    name: 'Michael Chen',
    role: 'CTO, InnovateNow',
    content: 'Outstanding web development and AI integration. The custom chatbot they built has revolutionized our customer service.',
    rating: 5,
  },
  {
    name: 'Emily Rodriguez',
    role: 'Founder, GrowthLab',
    content: 'Professional, innovative, and results-driven. They delivered exactly what we needed and exceeded our expectations.',
    rating: 5,
  },
  {
    name: 'David Thompson',
    role: 'Director, FutureTech',
    content: 'The automation solutions have saved us countless hours. Highly recommend their AI expertise and professional approach.',
    rating: 5,
  },
];

export default function ModernTestimonials({ className }: TestimonialsProps) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { ref: inViewRef, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  useEffect(() => {
    if (inView && sectionRef.current) {
      const tl = gsap.timeline();
      
      tl.fromTo('.testimonials-title', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
      )
      .fromTo('.testimonial-card', 
        { opacity: 0, y: 50, rotationY: 45 },
        { opacity: 1, y: 0, rotationY: 0, duration: 0.8, ease: 'power2.out', stagger: 0.2 },
        '-=0.4'
      );

      // Continuous floating animation for testimonial cards
      gsap.to('.testimonial-card', {
        y: -10,
        duration: 3,
        ease: 'power2.inOut',
        yoyo: true,
        repeat: -1,
        stagger: 0.5
      });

      // Moving background elements
      gsap.to('.moving-bg-element', {
        x: 100,
        y: 50,
        rotation: 360,
        duration: 20,
        ease: 'none',
        repeat: -1,
        stagger: 2
      });
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
      {/* Moving Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="moving-bg-element absolute top-10 left-10 w-20 h-20 bg-gradient-to-br from-[#A607F2]/20 to-purple-600/20 rounded-full blur-xl" />
        <div className="moving-bg-element absolute top-1/3 right-20 w-16 h-16 bg-gradient-to-br from-[#A607F2]/15 to-purple-600/15 rounded-full blur-lg" />
        <div className="moving-bg-element absolute bottom-20 left-1/4 w-24 h-24 bg-gradient-to-br from-[#A607F2]/10 to-purple-600/10 rounded-full blur-2xl" />
        <div className="moving-bg-element absolute bottom-1/3 right-10 w-12 h-12 bg-gradient-to-br from-[#A607F2]/25 to-purple-600/25 rounded-full blur-sm" />
      </div>

      {/* Border Gradients */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
      </div>
      
      <div ref={sectionRef} className="max-w-7xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="testimonials-title text-4xl md:text-6xl font-bold text-white mb-6">
            What Our <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#A607F2] to-purple-400">Clients Say</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Don't just take our word for it. Here's what our satisfied clients have to say about our AI solutions and web development services.
          </p>
        </div>

        <div ref={containerRef} className="grid md:grid-cols-2 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <div
              key={index}
              className="testimonial-card group relative perspective-1000"
            >
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border border-[#A607F2]/30 hover:border-[#A607F2]/60 transition-all duration-500 transform hover:scale-105 hover:shadow-2xl hover:shadow-[#A607F2]/20 h-full relative overflow-hidden">
                {/* Hover Effect Overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-[#A607F2]/5 to-purple-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl" />
                
                {/* Content */}
                <div className="relative z-10">
                  {/* Stars */}
                  <div className="flex gap-1 mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-5 h-5 fill-[#A607F2] text-[#A607F2]" />
                    ))}
                  </div>
                  
                  {/* Quote */}
                  <p className="text-gray-300 text-lg leading-relaxed mb-6 italic">
                    "{testimonial.content}"
                  </p>
                  
                  {/* Author */}
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-[#A607F2] to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      {testimonial.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div>
                      <h4 className="text-white font-semibold">{testimonial.name}</h4>
                      <p className="text-[#A607F2] text-sm">{testimonial.role}</p>
                    </div>
                  </div>
                </div>

                {/* Decorative Elements */}
                <div className="absolute top-4 right-4 w-8 h-8 bg-gradient-to-br from-[#A607F2]/20 to-purple-600/20 rounded-full opacity-50 group-hover:opacity-100 transition-opacity duration-300" />
                <div className="absolute bottom-4 left-4 w-6 h-6 bg-gradient-to-br from-[#A607F2]/15 to-purple-600/15 rounded-full opacity-30 group-hover:opacity-70 transition-opacity duration-300" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}