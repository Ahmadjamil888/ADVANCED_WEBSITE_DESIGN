"use client";

import { motion } from "framer-motion";
import { Target, ArrowRight } from "lucide-react";

export default function EarlyStageChaos() {
  return (
    <section className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-white/[0.02] to-black pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="relative"
        >
          {/* Main content container */}
          <div className="relative p-8 sm:p-12 lg:p-16 rounded-3xl bg-white/[0.02] border border-white/10 overflow-hidden">
            {/* Decorative gradient */}
            <div className="absolute top-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2 pointer-events-none" />

            {/* Icon */}
            <div className="flex justify-center mb-8">
              <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                <Target className="w-8 h-8 text-white/70" />
              </div>
            </div>

            {/* Slogan */}
            <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-light text-white text-center leading-tight mb-8">
              &ldquo;We&apos;ve lived the early-stage chaos —
              <br className="hidden sm:block" />
              now we help you build your way out of it.&rdquo;
            </h2>

            {/* Divider */}
            <div className="w-24 h-px bg-white/20 mx-auto mb-8" />

            {/* Supporting text */}
            <p className="text-base sm:text-lg text-white/50 text-center max-w-2xl mx-auto mb-10">
              From scrappy MVPs to enterprise-grade platforms, we understand the journey 
              because we&apos;ve walked it ourselves. Your vision deserves more than just code — 
              it deserves a partner who gets it.
            </p>

            {/* CTA Button */}
            <div className="flex justify-center">
              <a
                href="https://cal.com/zehanx-technologies-official"
                className="group inline-flex items-center gap-2 px-6 py-3 rounded-full text-sm text-white font-medium bg-white/5 backdrop-blur-md border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
              >
                Start Your Build
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
