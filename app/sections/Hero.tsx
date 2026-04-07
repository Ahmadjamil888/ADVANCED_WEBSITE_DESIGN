"use client";

import { motion } from "framer-motion";
import { Sparkles, Zap, Layers } from "lucide-react";
import VideoPlayer from "../components/VideoPlayer";
import LogoMarquee from "../components/LogoMarquee";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.3,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.16, 1, 0.3, 1] as const,
    },
  },
};

export default function Hero() {
  return (
    <section className="relative min-h-screen w-full bg-black overflow-hidden flex flex-col">
      {/* Background Video - positioned at bottom */}
      <div className="absolute bottom-[20vh] sm:bottom-[35vh] left-0 right-0 h-[50vh] sm:h-[60vh] lg:h-[80vh] w-full z-0">
        <VideoPlayer
          src="https://stream.mux.com/9JXDljEVWYwWu01PUkAemafDugK89o01BR6zqJ3aS9u00A.m3u8"
          className="w-full h-full object-cover"
        />
      </div>

      {/* Content Container */}
      <div className="relative z-10 flex flex-col items-center justify-center flex-1 px-4 pt-32 sm:pt-28 pb-32 sm:pb-24">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="flex flex-col items-center text-center max-w-5xl"
        >
          {/* Badges */}
          <motion.div variants={itemVariants} className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 mb-4 sm:mb-6">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10">
              <Sparkles className="w-3.5 h-3.5 text-white/70" />
              <span className="text-xs text-white/80">End-to-End Solutions</span>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10">
              <Zap className="w-3.5 h-3.5 text-white/70" />
              <span className="text-xs text-white/80">AI-Powered</span>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10">
              <Layers className="w-3.5 h-3.5 text-white/70" />
              <span className="text-xs text-white/80">From Idea to Deployment</span>
            </div>
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={itemVariants}
            className="text-4xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-5xl font-light text-white tracking-tight leading-[1.1] mb-3 sm:mb-4 text-center"
          >
            We Build What We Wish
            <br />
            We Had as Founders
          </motion.h1>

          {/* Positioning Statement */}
          <motion.p
            variants={itemVariants}
            className="text-base sm:text-lg lg:text-xl text-white/60 max-w-3xl mx-auto mb-6 sm:mb-8 leading-relaxed px-4 sm:px-0"
          >
            AI-powered SaaS, MVPs, and systems — built by a technical founder with real startup experience.
          </motion.p>

          {/* Buttons */}
          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center gap-2 sm:gap-3 w-full sm:w-auto px-4 sm:px-0">
            <a
              href="https://cal.com/zehanx-technologies-official"
              className="w-full sm:w-auto text-center px-5 sm:px-6 py-2.5 sm:py-3 rounded-full text-sm text-white font-medium bg-black border border-white/20 hover:border-white/40 transition-all duration-300 hover:shadow-lg hover:shadow-white/5"
            >
              Book a Call
            </a>
            <a
              href="https://cal.com/zehanx-technologies-official"
              className="w-full sm:w-auto text-center px-5 sm:px-6 py-2.5 sm:py-3 rounded-full text-sm text-white font-medium bg-white/5 backdrop-blur-md border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
            >
              Let&apos;s Build Together
            </a>
          </motion.div>
        </motion.div>
      </div>

      {/* Logo Marquee - positioned at bottom with proper spacing */}
      <div className="relative z-10 w-full mt-auto">
        <LogoMarquee />
      </div>
    </section>
  );
}
