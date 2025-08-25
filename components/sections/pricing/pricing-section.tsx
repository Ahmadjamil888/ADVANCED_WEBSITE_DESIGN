'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useInView } from 'react-intersection-observer';
import { Check, Star, Zap } from 'lucide-react';
import { Section } from '../../ui/section';
import { Button } from '../../ui/button';
import { cn } from '@/lib/utils';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

interface PricingSectionProps {
  className?: string;
}

const packages = [
  {
    name: 'Starter Package',
    price: '10,000',
    currency: 'PKR',
    icon: <Star className="w-8 h-8" />,
    features: [
      '1 Professional Website',
      '1 Custom Domain',
      'Custom Styling & Design',
      'Professional Content Writing',
      'AI Chatbot Integration',
      'Unlimited Revisions',
      '30 Days Support'
    ],
    popular: false,
  },
  {
    name: 'Professional Package',
    price: '25,000',
    currency: 'PKR',
    icon: <Zap className="w-8 h-8" />,
    features: [
      'Everything in Starter',
      'Advanced AI Features',
      'E-commerce Integration',
      'SEO Optimization',
      'Analytics Dashboard',
      'Social Media Integration',
      'Priority Support',
      '60 Days Maintenance'
    ],
    popular: true,
  },
  {
    name: 'Enterprise Package',
    price: 'Custom',
    currency: 'Pricing',
    icon: <Check className="w-8 h-8" />,
    features: [
      'Custom AI Solutions',
      'Advanced Automation',
      'Multi-platform Integration',
      'Dedicated Project Manager',
      'Custom Development',
      'Ongoing Maintenance',
      '24/7 Priority Support',
      'Scalable Architecture'
    ],
    popular: false,
  },
];

export default function PricingSection({ className }: PricingSectionProps) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const { ref: inViewRef, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  useEffect(() => {
    if (inView && sectionRef.current) {
      const tl = gsap.timeline();
      
      tl.fromTo('.pricing-title', 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
      )
      .fromTo('.pricing-card', 
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
        'py-20 bg-gradient-to-br from-gray-900 via-black to-gray-900 relative overflow-hidden',
        className
      )}
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#A607F2] to-transparent" />
      </div>
      
      <div ref={sectionRef} className="max-w-7xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="pricing-title text-4xl md:text-6xl font-bold text-white mb-6">
            Choose Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#A607F2] to-purple-400">Package</span>
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Tailored solutions for every business need. From startups to enterprises, we have the perfect package for you.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {packages.map((pkg, index) => (
            <div
              key={index}
              className={cn(
                'pricing-card group relative',
                pkg.popular && 'md:-mt-4'
              )}
            >
              {pkg.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-[#A607F2] to-purple-600 text-white px-6 py-2 rounded-full text-sm font-semibold">
                  Most Popular
                </div>
              )}
              
              <div className={cn(
                'bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-8 border transition-all duration-300 transform hover:scale-105 hover:shadow-2xl h-full',
                pkg.popular 
                  ? 'border-[#A607F2] shadow-lg shadow-[#A607F2]/20 hover:shadow-[#A607F2]/30' 
                  : 'border-gray-700 hover:border-[#A607F2]/50 hover:shadow-[#A607F2]/10'
              )}>
                <div className="text-center mb-8">
                  <div className={cn(
                    'inline-flex items-center justify-center w-16 h-16 rounded-full mb-4',
                    pkg.popular 
                      ? 'bg-gradient-to-br from-[#A607F2] to-purple-600 text-white' 
                      : 'bg-gray-700 text-gray-300'
                  )}>
                    {pkg.icon}
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-2">{pkg.name}</h3>
                  
                  <div className="mb-4">
                    <span className="text-4xl font-bold text-white">{pkg.price}</span>
                    <span className="text-gray-400 ml-2">{pkg.currency}</span>
                  </div>
                </div>

                <ul className="space-y-4 mb-8">
                  {pkg.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center gap-3">
                      <Check className="w-5 h-5 text-[#A607F2] flex-shrink-0" />
                      <span className="text-gray-300">{feature}</span>
                    </li>
                  ))}
                </ul>

                <Button
                  className={cn(
                    'w-full py-3 rounded-full font-semibold transition-all duration-300 transform hover:scale-105',
                    pkg.popular
                      ? 'bg-gradient-to-r from-[#A607F2] to-purple-600 hover:from-[#8A05D1] hover:to-purple-700 text-white shadow-lg hover:shadow-xl'
                      : 'bg-gray-700 hover:bg-[#A607F2] text-white'
                  )}
                  asChild
                >
                  <a href="/contact">Get In Touch</a>
                </Button>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center mt-12">
          <p className="text-gray-400 mb-4">
            Need something different? We create custom solutions for unique requirements.
          </p>
          <Button
            variant="outline"
            className="border-[#A607F2] text-[#A607F2] hover:bg-[#A607F2] hover:text-white px-8 py-3 rounded-full font-semibold transition-all duration-300"
            asChild
          >
            <a href="/contact">Contact Us for Custom Quote</a>
          </Button>
        </div>
      </div>
    </Section>
  );
}