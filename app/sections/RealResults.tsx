"use client";

import { motion } from "framer-motion";
import { TrendingUp, Clock, Database, Shield, Rocket, BarChart3 } from "lucide-react";

const caseStudies = [
  {
    icon: Shield,
    metric: "37%",
    metricLabel: "Fraud Reduction",
    title: "Built fraud detection system",
    description: "Reduced losses by 37%",
    detail: "AI-powered real-time fraud detection with automated risk scoring",
  },
  {
    icon: Rocket,
    metric: "21",
    metricLabel: "Days to MVP",
    title: "Launched SaaS MVP",
    description: "From concept to market in 21 days",
    detail: "Complete platform with auth, dashboard, payments, and deployment",
  },
  {
    icon: Database,
    metric: "1M+",
    metricLabel: "Records Handled",
    title: "Scaled enterprise dashboard",
    description: "Handling 1M+ records seamlessly",
    detail: "High-performance data visualization with sub-second query response",
  },
  {
    icon: TrendingUp,
    metric: "3x",
    metricLabel: "Efficiency Gain",
    title: "Automated operations workflow",
    description: "3x faster processing",
    detail: "End-to-end automation replacing manual Excel-based operations",
  },
  {
    icon: BarChart3,
    metric: "85%",
    metricLabel: "Prediction Accuracy",
    title: "Deployed ML prediction engine",
    description: "85% accuracy on forecasts",
    detail: "Custom ML models for demand forecasting and inventory optimization",
  },
  {
    icon: Clock,
    metric: "60%",
    metricLabel: "Time Saved",
    title: "AI-powered customer support",
    description: "60% reduction in response time",
    detail: "Intelligent chatbot handling 80% of common queries automatically",
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

export default function RealResults() {
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
            <TrendingUp className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Proof of Work</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Real Results
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Metrics that matter. Outcomes that drive business growth.
          </p>
        </motion.div>

        {/* Case Studies Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8"
        >
          {caseStudies.map((study) => (
            <motion.div
              key={study.title}
              variants={itemVariants}
              className="group relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              {/* Metric Badge */}
              <div className="inline-flex flex-col items-start mb-4">
                <span className="text-3xl sm:text-4xl font-light text-white">
                  {study.metric}
                </span>
                <span className="text-xs text-white/40 uppercase tracking-wider">
                  {study.metricLabel}
                </span>
              </div>

              {/* Icon */}
              <div className="absolute top-6 right-6 w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                <study.icon className="w-5 h-5 text-white/60" />
              </div>

              {/* Title & Description */}
              <h3 className="text-lg sm:text-xl font-light text-white mb-1">
                {study.title}
              </h3>
              <p className="text-sm text-white/70 mb-3">
                {study.description}
              </p>

              {/* Detail */}
              <p className="text-sm text-white/40 leading-relaxed">
                {study.detail}
              </p>
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
            <span className="text-white/80">6+ years experience</span> building systems for startups and enterprises. 
            <span className="text-white/80"> Trusted by businesses worldwide</span> to deliver results that move the needle.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
