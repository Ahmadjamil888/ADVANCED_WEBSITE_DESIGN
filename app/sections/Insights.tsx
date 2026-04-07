"use client";

import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";
import { TrendingUp, Users, Globe, Clock, Star, Quote } from "lucide-react";

const stats = [
  {
    icon: TrendingUp,
    value: 6,
    suffix: "+",
    label: "Years Experience",
    description: "Trusted service since 2018",
  },
  {
    icon: Users,
    value: 100,
    suffix: "+",
    label: "Projects Delivered",
    description: "Successful completions",
  },
  {
    icon: Globe,
    value: 50,
    suffix: "+",
    label: "Happy Clients",
    description: "Worldwide partnerships",
  },
  {
    icon: Clock,
    value: 24,
    suffix: "/7",
    label: "Support",
    description: "Always available",
  },
];

const testimonials = [
  {
    quote: "zehanx Technologies transformed our business with their AI solutions. The team's expertise and dedication exceeded our expectations.",
    author: "Sarah Johnson",
    role: "CTO, TechStart Inc.",
    rating: 5,
  },
  {
    quote: "Outstanding web development services. They delivered our platform on time and the quality was exceptional. Highly recommended!",
    author: "Michael Chen",
    role: "Founder, GrowthLabs",
    rating: 5,
  },
  {
    quote: "The machine learning models they built for us have significantly improved our predictive analytics capabilities.",
    author: "David Rodriguez",
    role: "Data Director, FinanceHub",
    rating: 5,
  },
];

const clientIndustries = [
  "Healthcare",
  "Finance",
  "Education",
  "E-Commerce",
  "Real Estate",
  "Logistics",
  "Technology",
  "Manufacturing",
];

function AnimatedCounter({ value, suffix }: { value: number; suffix: string }) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (latest) => Math.round(latest));

  useEffect(() => {
    const controls = animate(count, value, {
      duration: 2,
      ease: "easeOut",
    });
    return controls.stop;
  }, [count, value]);

  return (
    <span className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white tabular-nums">
      <motion.span>{rounded}</motion.span>
      {suffix}
    </span>
  );
}

export default function Insights() {
  return (
    <section id="insights" className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-white/[0.02] to-black pointer-events-none" />
      
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
            <TrendingUp className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Insights</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-semibold text-white tracking-tight mb-4 sm:mb-6">
            Trusted by businesses
            <br className="hidden sm:block" />
            worldwide
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Numbers that reflect our commitment. See why clients choose zehanx Technologies for their digital transformation.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 lg:gap-8 mb-20 sm:mb-32">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                duration: 0.6, 
                delay: index * 0.1,
                ease: [0.16, 1, 0.3, 1] as const 
              }}
              className="relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 text-center"
            >
              <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-white/5 mx-auto mb-4 sm:mb-6">
                <stat.icon className="w-5 h-5 sm:w-6 sm:h-6 text-white/80" />
              </div>
              <AnimatedCounter value={stat.value} suffix={stat.suffix} />
              <h3 className="text-base sm:text-lg font-medium text-white mt-2 sm:mt-4 mb-1">
                {stat.label}
              </h3>
              <p className="text-sm text-white/50">
                {stat.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Industries We Serve */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Globe className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Industries We Serve</span>
          </div>
          <h3 className="text-2xl sm:text-3xl lg:text-4xl font-light text-white tracking-tight mb-4">
            Diverse expertise across sectors
          </h3>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="flex flex-wrap justify-center gap-3 sm:gap-4 mb-20 sm:mb-32"
        >
          {clientIndustries.map((industry, index) => (
            <motion.span
              key={industry}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="px-4 sm:px-6 py-2 sm:py-3 rounded-full bg-white/[0.02] border border-white/10 text-sm sm:text-base text-white/70 hover:bg-white/[0.04] hover:border-white/20 transition-all cursor-default"
            >
              {industry}
            </motion.span>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
