"use client";

import { motion } from "framer-motion";
import { ExternalLink, Code, Brain, Bot, Network, Smartphone, Globe } from "lucide-react";

const projects = [
  {
    title: "AI-Powered Analytics Platform",
    description: "Enterprise-grade analytics dashboard with machine learning predictions and real-time data visualization.",
    tags: ["AI/ML", "React", "Python"],
    icon: Brain,
    gradient: "from-blue-500/20 via-purple-500/20 to-pink-500/20",
  },
  {
    title: "E-Commerce Mobile App",
    description: "Full-featured shopping application with AR product preview and smart recommendation engine.",
    tags: ["Mobile", "React Native", "Node.js"],
    icon: Smartphone,
    gradient: "from-green-500/20 via-teal-500/20 to-cyan-500/20",
  },
  {
    title: "Neural Network Trading Bot",
    description: "Deep learning system for automated trading with 99.9% accuracy in pattern recognition.",
    tags: ["Deep Learning", "TensorFlow", "AWS"],
    icon: Network,
    gradient: "from-orange-500/20 via-red-500/20 to-pink-500/20",
  },
  {
    title: "Healthcare Management System",
    description: "Comprehensive patient management platform with AI-assisted diagnosis capabilities.",
    tags: ["Healthcare", "AI", "Full-Stack"],
    icon: Bot,
    gradient: "from-cyan-500/20 via-blue-500/20 to-indigo-500/20",
  },
  {
    title: "Social Media Dashboard",
    description: "Unified analytics platform for managing multiple social accounts with sentiment analysis.",
    tags: ["Web App", "NLP", "React"],
    icon: Globe,
    gradient: "from-purple-500/20 via-pink-500/20 to-rose-500/20",
  },
  {
    title: "IoT Smart Home Platform",
    description: "Connected device ecosystem with voice control and predictive automation algorithms.",
    tags: ["IoT", "AI", "Embedded"],
    icon: Code,
    gradient: "from-emerald-500/20 via-green-500/20 to-lime-500/20",
  },
];

// Animated floating particles for cards
const FloatingParticles = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <div className="absolute top-4 left-4 w-2 h-2 rounded-full bg-white/20 animate-pulse" />
    <div className="absolute top-8 right-6 w-1.5 h-1.5 rounded-full bg-white/10 animate-pulse delay-75" />
    <div className="absolute bottom-6 left-8 w-1 h-1 rounded-full bg-white/15 animate-pulse delay-150" />
    <div className="absolute bottom-4 right-4 w-2 h-2 rounded-full bg-white/10 animate-pulse delay-300" />
  </div>
);

// Animated gradient orb
const GradientOrb = ({ gradient }: { gradient: string }) => (
  <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full bg-gradient-to-br ${gradient} blur-3xl opacity-40 group-hover:opacity-60 transition-opacity duration-500`} />
);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
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

export default function Portfolio() {
  return (
    <section id="portfolio" className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-white/[0.01] to-black pointer-events-none" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-12 sm:mb-16 lg:mb-20"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Code className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Portfolio</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Our Recent
            <br className="hidden sm:block" />
            Projects
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Explore our latest work across web development, AI, machine learning, and mobile applications.
          </p>
        </motion.div>

        {/* Projects Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8"
        >
          {projects.map((project) => (
            <motion.div
              key={project.title}
              variants={itemVariants}
              className="group relative"
            >
              {/* Card with black background and border styling */}
              <div className="relative p-6 sm:p-8 rounded-xl border border-[rgba(255,255,255,0.10)] bg-[#000000] shadow-[2px_4px_16px_0px_rgba(248,248,248,0.06)_inset] overflow-hidden transition-all duration-300 hover:border-[rgba(255,255,255,0.20)]">
                <FloatingParticles />
                
                {/* Gradient Orb Background */}
                <GradientOrb gradient={project.gradient} />
                
                {/* Top visual area with icon circles */}
                <div className="relative h-[12rem] sm:h-[15rem] rounded-xl z-10 mb-6 overflow-hidden">
                  {/* Background glow */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${project.gradient} opacity-30`} />
                  
                  {/* Animated circles with icons */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="flex flex-row flex-shrink-0 justify-center items-center gap-3">
                      {/* Small circle */}
                      <div className="rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)] shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)] h-10 w-10">
                        <div className={`w-4 h-4 rounded-full bg-gradient-to-br ${project.gradient} opacity-60`} />
                      </div>
                      
                      {/* Medium circle with icon */}
                      <div className="rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)] shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)] h-14 w-14">
                        <project.icon className="w-6 h-6 text-white/70" />
                      </div>
                      
                      {/* Large circle */}
                      <div className="rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)] shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)] h-20 w-20">
                        <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${project.gradient} opacity-40 blur-sm`} />
                      </div>
                      
                      {/* Medium circle */}
                      <div className="rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)] shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)] h-14 w-14">
                        <div className={`w-6 h-6 rounded-full bg-gradient-to-br ${project.gradient} opacity-50`} />
                      </div>
                      
                      {/* Small circle */}
                      <div className="rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)] shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)] h-10 w-10">
                        <div className={`w-3 h-3 rounded-full bg-gradient-to-br ${project.gradient} opacity-70`} />
                      </div>
                    </div>
                  </div>
                  
                  {/* Animated scan line */}
                  <div className="absolute top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-white/30 to-transparent animate-pulse left-1/2 -translate-x-1/2" />
                </div>

                {/* External Link - appears on hover */}
                <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity z-20">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-white/10 backdrop-blur-md border border-white/20 hover:bg-white/20 transition-colors cursor-pointer">
                    <ExternalLink className="w-4 h-4 text-white/80" />
                  </div>
                </div>

                {/* Content */}
                <div className="relative z-10">
                  <h3 className="text-lg font-light text-white mb-2 group-hover:text-white/90 transition-colors">
                    {project.title}
                  </h3>
                  <p className="text-sm text-white/50 mb-4 leading-relaxed">
                    {project.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {project.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-3 py-1 text-xs text-white/60 bg-white/5 rounded-full border border-white/10"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4, ease: [0.16, 1, 0.3, 1] as const }}
          className="mt-12 sm:mt-16 text-center"
        >
          <a
            href="#contact"
            className="inline-flex items-center gap-2 px-6 sm:px-8 py-3 sm:py-4 rounded-full text-white font-light bg-white/5 backdrop-blur-md border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
          >
            Start Your Project
            <ExternalLink className="w-4 h-4" />
          </a>
        </motion.div>
      </div>
    </section>
  );
}
