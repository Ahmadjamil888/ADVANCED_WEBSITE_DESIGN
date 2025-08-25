'use client';

import { cn } from '@/lib/utils';

interface ModernHeroProps {
  className?: string;
}

export default function ModernHero({ className }: ModernHeroProps) {
  return (
    <section 
      className={cn(
        "relative min-h-screen flex items-center justify-center pt-16",
        "bg-black",
        className
      )}
    >
      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Badge */}
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-blue-500/30 text-sm font-medium text-blue-300 shadow-lg shadow-blue-500/10">
            <div className="w-2 h-2 bg-blue-400 rounded-full" />
            <span>AI-Powered Solutions</span>
          </div>
        </div>
        
        {/* Main Title */}
        <h1 className="mb-6">
          <span className="block text-5xl sm:text-6xl lg:text-7xl font-bold text-white mb-4 leading-tight">
            Transform Your Business
          </span>
          <span className="block text-4xl sm:text-5xl lg:text-6xl font-bold text-blue-300 leading-tight">
            with AI & Innovation
          </span>
        </h1>
        
        {/* Description */}
        <p className="text-xl text-white max-w-3xl mx-auto mb-10 leading-relaxed">
          We build intelligent solutions that drive real results. From AI automation to modern web applications, 
          we help businesses thrive in the digital age.
        </p>
        
        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <a
            href="/contact"
            className="inline-flex items-center gap-2 px-8 py-4 bg-blue-600 text-white font-semibold rounded-lg shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-shadow duration-300"
          >
            <span>Start Your Project</span>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </a>
          
          <a
            href="/portfolio"
            className="inline-flex items-center gap-2 px-8 py-4 border border-blue-500/30 text-white font-semibold rounded-lg shadow-lg shadow-blue-500/10 hover:shadow-blue-500/20 transition-shadow duration-300"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <span>View Our Work</span>
          </a>
        </div>
        
        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
          {[
            { number: "100+", label: "Projects Delivered" },
            { number: "99%", label: "Client Satisfaction" },
            { number: "24/7", label: "Support Available" },
            { number: "5+", label: "Years Experience" }
          ].map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl font-bold text-white mb-2">{stat.number}</div>
              <div className="text-sm text-blue-300">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
        <div className="flex flex-col items-center gap-2 text-blue-300">
          <span className="text-xs">Scroll to explore</span>
          <div className="w-5 h-8 border border-blue-300 rounded-full flex justify-center">
            <div className="w-1 h-2 bg-blue-300 rounded-full mt-2" />
          </div>
        </div>
      </div>
    </section>
  );
}