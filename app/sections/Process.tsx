"use client";

import { motion } from "framer-motion";
import { Search, Lightbulb, Code2, Rocket, ArrowRight } from "lucide-react";

const steps = [
  {
    number: "01",
    icon: Search,
    title: "Discovery",
    description: "We analyze your requirements, understand your goals, and define the project scope to ensure we deliver exactly what you need.",
  },
  {
    number: "02",
    icon: Lightbulb,
    title: "Strategy",
    description: "Our team designs a comprehensive roadmap, selecting the right technologies and methodologies for your project.",
  },
  {
    number: "03",
    icon: Code2,
    title: "Development",
    description: "We build your solution with clean code, following best practices in AI, ML, web, and mobile development.",
  },
  {
    number: "04",
    icon: Rocket,
    title: "Deployment",
    description: "Your project goes live with thorough testing, documentation, and ongoing support for long-term success.",
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
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

export default function Process() {
  return (
    <section id="process" className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-white/[0.01] to-black pointer-events-none" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-16 sm:mb-20 lg:mb-24"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Rocket className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Our Process</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            How We Work
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Our proven 4-step process ensures every project is delivered with precision, quality, and excellence.
          </p>
        </motion.div>

        {/* Process Steps */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8"
        >
          {steps.map((step, index) => (
            <motion.div
              key={step.number}
              variants={itemVariants}
              className="group relative"
            >
              {/* Connector line for desktop */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-16 left-[60%] w-[80%] h-[1px] bg-gradient-to-r from-white/20 to-transparent" />
              )}
              
              <div className="relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300 h-full">
                {/* Step Number */}
                <div className="absolute -top-3 -left-2 sm:-top-4 sm:-left-3">
                  <span className="text-5xl sm:text-6xl font-thin text-white/10 group-hover:text-white/20 transition-colors">
                    {step.number}
                  </span>
                </div>

                {/* Icon */}
                <div className="relative mb-6">
                  <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-white/5 border border-white/10 group-hover:bg-white/10 transition-colors">
                    <step.icon className="w-6 h-6 text-white/80" />
                  </div>
                </div>

                {/* Content */}
                <h3 className="text-xl sm:text-2xl font-light text-white mb-3">
                  {step.title}
                </h3>
                <p className="text-sm sm:text-base text-white/60 leading-relaxed font-light">
                  {step.description}
                </p>

                {/* Arrow indicator */}
                <div className="mt-6 flex items-center gap-2 text-white/30 group-hover:text-white/60 transition-colors">
                  <span className="text-sm font-light">Learn more</span>
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Bottom Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="mt-16 sm:mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
        >
          <div>
            <div className="text-3xl sm:text-4xl font-light text-white mb-2">100%</div>
            <div className="text-sm text-white/50">Client Satisfaction</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-light text-white mb-2">6+</div>
            <div className="text-sm text-white/50">Years Experience</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-light text-white mb-2">100+</div>
            <div className="text-sm text-white/50">Projects Completed</div>
          </div>
          <div>
            <div className="text-3xl sm:text-4xl font-light text-white mb-2">24/7</div>
            <div className="text-sm text-white/50">Support Available</div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
