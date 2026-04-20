"use client";

import { motion } from "framer-motion";
import { ExternalLink, Code, Brain, Bot, Network, Smartphone, Globe } from "lucide-react";

const projects = [
  {
    client: "Dr. Saira",
    role: "CEO, Usman Hospital",
    title: "BrainSkills AI",
    subtitle: "AI Teaching & Learning Platform",
    description: "Complete learning management system with AI-powered content generation, student progress tracking, and automated assessments. Built for medical education.",
    outcome: "Reduced course prep time by 60%",
    tags: ["EdTech", "AI/ML", "LMS"],
    icon: Brain,
    gradient: "from-blue-500/20 via-purple-500/20 to-pink-500/20",
  },
  {
    client: "Rana Asif Khan",
    role: "CEO, IRTCoP",
    title: "Institute Management System",
    subtitle: "Operations & Student Portal",
    description: "End-to-end institute management with student enrollment, course scheduling, fee tracking, and certificate generation. Handles 500+ active students.",
    outcome: "Admin workload reduced by 70%",
    tags: ["Education", "Full-Stack", "Automation"],
    icon: Globe,
    gradient: "from-green-500/20 via-teal-500/20 to-cyan-500/20",
  },
  {
    client: "Umair Fiaz",
    role: "CEO, Janjua Tailors",
    title: "OrderFlow AI",
    subtitle: "Autonomous Fulfillment Engine",
    description: "Custom AI-driven fulfillment system with autonomous measurement extraction, automated delivery scheduling, and AI customer notifications.",
    outcome: "Order processing 3x faster",
    tags: ["AI Ops", "Automation", "Workflow"],
    icon: Zap,
    gradient: "from-orange-500/20 via-red-500/20 to-pink-500/20",
  },
  {
    client: "Syeda Eyesha Nadeem",
    role: "CEO, APS Jinnah",
    title: "SchoolSync",
    subtitle: "School Administration System",
    description: "Comprehensive school management with attendance, gradebooks, parent portal, and automated reporting. Serves 800+ students and staff.",
    outcome: "Parent communication improved 80%",
    tags: ["Education", "SaaS", "Portal"],
    icon: Bot,
    gradient: "from-cyan-500/20 via-blue-500/20 to-indigo-500/20",
  },
  {
    client: "Shazab Jamil",
    role: "CEO, Daak Khana",
    title: "DocuSense AI",
    subtitle: "Vision-AI Sorting Platform",
    description: "Real-time package recognition with route optimization and autonomous document processing. Processes 10,000+ packages monthly with custom fine-tuned models.",
    outcome: "Data accuracy up to 98%",
    tags: ["Vision-AI", "Logistics", "LLM"],
    icon: Network,
    gradient: "from-purple-500/20 via-pink-500/20 to-rose-500/20",
  },
  {
    client: "Internal Product",
    role: "Zehanx Labs",
    title: "DataFlow Analytics",
    subtitle: "Business Intelligence Dashboard",
    description: "Enterprise analytics processing 1M+ records with predictive forecasting and automated reporting. Sub-second query response times.",
    outcome: "Decision-making speed +40%",
    tags: ["Analytics", "Predictive", "SaaS"],
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
            Trusted by Founders
            <br className="hidden sm:block" />
            & Business Owners
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Real systems built for real businesses. From hospitals to schools to logistics — we build what moves the needle.
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

                {/* Client Badge */}
                <div className="relative z-10 mb-4">
                  <p className="text-xs text-white/40 uppercase tracking-wider">{project.client}</p>
                  <p className="text-xs text-white/30">{project.role}</p>
                </div>

                {/* Content */}
                <div className="relative z-10">
                  <h3 className="text-lg font-light text-white mb-1 group-hover:text-white/90 transition-colors">
                    {project.title}
                  </h3>
                  <p className="text-sm text-white/60 mb-2">{project.subtitle}</p>
                  <p className="text-sm text-white/50 mb-3 leading-relaxed">
                    {project.description}
                  </p>
                  <p className="text-sm text-white/80 font-medium mb-4">
                    {project.outcome}
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

        {/* SaaS-Style Case Study Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4, ease: [0.16, 1, 0.3, 1] as const }}
          className="mt-16 sm:mt-24"
        >
          <div className="p-8 sm:p-12 rounded-3xl bg-white/[0.02] border border-white/10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 mb-4">
                  <span className="text-xs text-white/60">SaaS Case Study</span>
                </div>
                <h3 className="text-2xl sm:text-3xl font-light text-white mb-4">
                  Multi-Agent Marketing Engine
                </h3>
                <p className="text-sm sm:text-base text-white/60 mb-6 leading-relaxed">
                  Built a complete autonomous marketing SaaS that researches, writes, and distributes content across 5 channels without human intervention.
                </p>
                <div className="flex flex-wrap gap-4 mb-6">
                  <div>
                    <p className="text-2xl font-light text-white">21 Days</p>
                    <p className="text-xs text-white/40">To MVP</p>
                  </div>
                  <div>
                    <p className="text-2xl font-light text-white">100+</p>
                    <p className="text-xs text-white/40">Paying Customers</p>
                  </div>
                  <div>
                    <p className="text-2xl font-light text-white">$5k–$25k+</p>
                    <p className="text-xs text-white/40">Project Value</p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <p className="text-sm text-white/80 mb-1">Problem</p>
                  <p className="text-xs text-white/50">Manual reporting taking 40 hours/week</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <p className="text-sm text-white/80 mb-1">Solution</p>
                  <p className="text-xs text-white/50">Automated real-time analytics dashboard</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <p className="text-sm text-white/80 mb-1">Tech Stack</p>
                  <p className="text-xs text-white/50">React, Node.js, PostgreSQL, AWS</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <p className="text-sm text-white/80 mb-1">Result</p>
                  <p className="text-xs text-white/50">Reporting time reduced to 2 hours/week</p>
                </div>
              </div>
            </div>
          </div>
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
            href="https://cal.com/zehanx-technologies-official"
            className="inline-flex items-center gap-2 px-6 sm:px-8 py-3 sm:py-4 rounded-full text-white font-light bg-white/5 backdrop-blur-md border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
          >
            Build Your System
            <ExternalLink className="w-4 h-4" />
          </a>
        </motion.div>
      </div>
    </section>
  );
}
