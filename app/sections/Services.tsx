"use client";

import { motion } from "framer-motion";
import { Code, Brain, Bot, Network, Smartphone, Globe, ArrowRight, Zap, Star, Rocket, Database } from "lucide-react";

const primaryService = {
  icon: Brain,
  badge: "Core Offer ($5k–$25k+)",
  title: "AI-Powered SaaS Development",
  description: "Complete business systems from idea to deployment. You get a revenue-ready platform with everything needed to start acquiring customers.",
  features: ["Landing Page & Marketing", "Admin + User Dashboards", "Auth & User Roles", "Database & APIs", "AI Features (Optional)", "Deployment & Domain"],
};

const secondaryServices = [
  {
    icon: Rocket,
    title: "21-Day MVP System",
    description: "From concept to paying customers. We validate, build, and ship fast so you can test your market quickly.",
    highlight: "Concept → Live in 21 Days",
  },
  {
    icon: Database,
    title: "AI Intelligence Layer",
    description: "Add predictive analytics, automation, and smart features to your existing platform. $5k–$25k add-on.",
    highlight: "Upsell: $5k–$25k",
  },
  {
    icon: Zap,
    title: "Optimize & Scale",
    description: "Ongoing optimization, analytics setup, and continuous improvement. For founders scaling from 1→10.",
    highlight: "Retainer: $2k–$10k/mo",
  },
];

const otherServices = [
  {
    icon: Code,
    title: "Web Development",
    description: "Custom web applications",
  },
  {
    icon: Smartphone,
    title: "App Development",
    description: "Mobile apps for iOS & Android",
  },
  {
    icon: Globe,
    title: "Enterprise Software",
    description: "Custom business solutions",
  },
];

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

export default function Services() {
  return (
    <section id="services" className="relative w-full bg-black py-20 sm:py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
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
            <span className="text-sm text-white/80">AI Product Engineering Partner</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Product Engineering
            <br className="hidden sm:block" />
            for SaaS Founders
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Not a dev shop. A technical partner who ships revenue-ready systems in 21 days. Projects range $5k–$25k+ depending on scope.
          </p>
        </motion.div>

        {/* Primary Service - BIG */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="mb-8 sm:mb-12"
        >
          <div className="relative p-8 sm:p-12 lg:p-16 rounded-3xl bg-white/[0.03] border border-white/20 hover:border-white/30 transition-all duration-300">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 border border-white/20 mb-6">
              <Star className="w-3.5 h-3.5 text-white/70" />
              <span className="text-xs text-white/80 font-medium">{primaryService.badge}</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
              {/* Left: Title & Description */}
              <div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <primaryService.icon className="w-7 h-7 text-white/80" />
                  </div>
                </div>
                <h3 className="text-2xl sm:text-3xl lg:text-4xl font-light text-white mb-4">
                  {primaryService.title}
                </h3>
                <p className="text-base sm:text-lg text-white/60 leading-relaxed mb-6">
                  {primaryService.description}
                </p>
                <a
                  href="https://cal.com/zehanx-technologies-official"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm text-white font-medium bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
                >
                  Book Strategy Call
                  <ArrowRight className="w-4 h-4" />
                </a>
              </div>

              {/* Right: Features Grid */}
              <div className="grid grid-cols-2 gap-4">
                {primaryService.features.map((feature, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-3 p-4 rounded-xl bg-white/[0.02] border border-white/10"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-white/40" />
                    <span className="text-sm text-white/70">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Secondary Services */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
        >
          {secondaryServices.map((service) => (
            <motion.div
              key={service.title}
              variants={itemVariants}
              className="group relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                  <service.icon className="w-5 h-5 text-white/70" />
                </div>
                <span className="text-xs text-white/40 px-2 py-1 rounded-full bg-white/5 border border-white/10">
                  {service.highlight}
                </span>
              </div>
              <h3 className="text-lg sm:text-xl font-light text-white mb-2">
                {service.title}
              </h3>
              <p className="text-sm text-white/50 leading-relaxed">
                {service.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Other Services - Compact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] as const }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-4"
        >
          {otherServices.map((service) => (
            <div
              key={service.title}
              className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.01] border border-white/5 hover:bg-white/[0.03] transition-all duration-300"
            >
              <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                <service.icon className="w-4 h-4 text-white/50" />
              </div>
              <div>
                <h4 className="text-sm font-light text-white/80">{service.title}</h4>
                <p className="text-xs text-white/40">{service.description}</p>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
