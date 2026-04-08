"use client";

import { motion } from "framer-motion";
import { Rocket, Clock, Users, Target, Zap, Shield } from "lucide-react";

const reasons = [
  {
    icon: Rocket,
    title: "21-Day Delivery",
    description: "Most agencies take 3-6 months. We ship production-ready systems in 21 days. Speed is our competitive advantage.",
  },
  {
    icon: Users,
    title: "Founder-First Thinking",
    description: "Built 3 startups ourselves. We understand the pressure of runway, investor expectations, and the need to ship yesterday.",
  },
  {
    icon: Target,
    title: "Business Outcomes, Not Code",
    description: "We do not just write features. We build systems that drive revenue, reduce costs, and scale with your growth.",
  },
  {
    icon: Zap,
    title: "AI-Native Architecture",
    description: "AI is not an afterthought. We architect systems where intelligence is core to the product, not a bolt-on.",
  },
  {
    icon: Clock,
    title: "Zero Management Overhead",
    description: "You get a complete team — product thinking, design, engineering, deployment. One point of contact, full accountability.",
  },
  {
    icon: Shield,
    title: "$10k–$50k Predictable Pricing",
    description: "No hourly billing surprises. Fixed-price projects with clear deliverables. You know exactly what you are paying for.",
  },
];

export default function WhyUs() {
  return (
    <section className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
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
            <Zap className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Why Zehanx</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Not a Dev Shop.
            <br className="hidden sm:block" />
            <span className="text-white/60">A Technical Partner.</span>
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Six reasons founders choose us over traditional agencies.
          </p>
        </motion.div>

        {/* Reasons Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          {reasons.map((reason, index) => (
            <motion.div
              key={reason.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{
                duration: 0.6,
                delay: index * 0.1,
                ease: [0.16, 1, 0.3, 1] as const,
              }}
              className="group relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              {/* Icon */}
              <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-4 group-hover:bg-white/10 transition-colors">
                <reason.icon className="w-6 h-6 text-white/70" />
              </div>

              {/* Content */}
              <h3 className="text-lg sm:text-xl font-light text-white mb-3">
                {reason.title}
              </h3>
              <p className="text-sm text-white/50 leading-relaxed">
                {reason.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-16 sm:mt-20 text-center"
        >
          <a
            href="https://cal.com/zehanx-technologies-official"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-full text-white font-light bg-white/5 backdrop-blur-md border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
          >
            Book a Strategy Call
          </a>
          <p className="mt-4 text-sm text-white/40">
            2 spots available this month
          </p>
        </motion.div>
      </div>
    </section>
  );
}
