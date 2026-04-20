"use client";

import { motion } from "framer-motion";
import { Briefcase, Zap, BarChart3 } from "lucide-react";

const offerTiers = [
  {
    icon: Briefcase,
    number: "1",
    title: "Full App Platform",
    subtitle: "Anchor Offer — $15k to $40k+",
    description: "Build a premium standalone mobile app or headless store to completely own your audience.",
    features: [
      "Custom Native Mobile App (iOS/Android)",
      "High-performance Headless Commerce",
      "Advanced customer retention portals",
      "AI-powered product recommendations",
      "Premium brand experience & max LTV",
    ],
    cta: "Outcome: Premium Brand Authority",
  },
  {
    icon: Zap,
    number: "2",
    title: "Conversion Leak Fix",
    subtitle: "Entry Offer — $299 (Normally $1k+)",
    description: "We find and fix 2–3 conversion leaks costing you sales right now. Limited to first 5 brands.",
    features: [
      "1 high-impact feature (e.g. abandoned cart recovery)",
      "3 conversion UX fixes mapped out",
      "Quick implementation or prototype",
      "Checkout bottleneck analysis",
      "Delivered in 5 days",
    ],
    cta: "Outcome: Immediate ROI & Recovered Sales",
  },
  {
    icon: BarChart3,
    number: "3",
    title: "System Expansion",
    subtitle: "Mid Tier — $1k to $3k",
    description: "Add powerful, revenue-generating features or optimize your existing e-commerce flow.",
    features: [
      "One-click upsell flows",
      "Checkout optimization",
      "COD confirmation automation",
      "Advanced revenue tracking",
      "Priority feature deployment",
    ],
    cta: "Outcome: Automated Revenue Increases",
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

export default function Offers() {
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
            <Briefcase className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Our Offer Structure</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            How We Work
            <br className="hidden sm:block" />
            Together
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Three ways to partner with us. Each designed for different stages of your growth.
          </p>
          <p className="text-sm sm:text-base text-white/40 max-w-xl mx-auto mt-4">
            You are not buying code. You are buying outcomes.
          </p>
        </motion.div>

        {/* Offer Tiers */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8"
        >
          {offerTiers.map((tier) => (
            <motion.div
              key={tier.number}
              variants={itemVariants}
              className="group relative"
            >
              <div className="relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300 h-full">
                {/* Number badge */}
                <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-white/10 border border-white/20 flex items-center justify-center text-sm font-medium text-white/80">
                  {tier.number}
                </div>

                {/* Icon */}
                <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-4">
                  <tier.icon className="w-6 h-6 text-white/70" />
                </div>

                {/* Subtitle */}
                <p className="text-xs text-white/40 uppercase tracking-wider mb-2">
                  {tier.subtitle}
                </p>

                {/* Title */}
                <h3 className="text-xl sm:text-2xl font-light text-white mb-3">
                  {tier.title}
                </h3>

                {/* Description */}
                <p className="text-sm text-white/50 mb-6">
                  {tier.description}
                </p>

                {/* Features */}
                <ul className="space-y-3 mb-6">
                  {tier.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <span className="w-1.5 h-1.5 rounded-full bg-white/30 mt-2 flex-shrink-0" />
                      <span className="text-sm text-white/60">{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                <p className="text-sm text-white/80 font-medium pt-4 border-t border-white/10">
                  {tier.cta}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Trust Statement */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] as const }}
          className="mt-16 sm:mt-20 text-center"
        >
          <p className="text-base sm:text-lg text-white/50 max-w-2xl mx-auto">
            <span className="text-white/80">Built 3 startups</span> and learned what founders actually need. 
            <span className="text-white/80"> 50+ systems shipped</span> for founders who value speed and quality.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
