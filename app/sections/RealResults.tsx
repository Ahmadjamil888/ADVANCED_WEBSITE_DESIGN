"use client";

import { motion } from "framer-motion";
import { TrendingUp, Clock, Database, Shield, Rocket, BarChart3 } from "lucide-react";

const caseStudies = [
  {
    icon: Shield,
    metric: "60%",
    metricLabel: "Course Prep Time Saved",
    title: "BrainSkills AI for Dr. Saira",
    description: "AI-powered medical education platform",
    detail: "Complete LMS with AI content generation serving 200+ medical students at Usman Hospital",
  },
  {
    icon: Rocket,
    metric: "21",
    metricLabel: "Days to MVP",
    title: "SaaS Analytics Platform",
    description: "From concept to 100 paying customers",
    detail: "Full-stack platform with auth, dashboards, payments, and AI-powered insights",
  },
  {
    icon: Database,
    metric: "500+",
    metricLabel: "Students Managed",
    title: "IRTCoP Institute System",
    description: "End-to-end education platform",
    detail: "Enrollment, scheduling, fees, and certificates — admin workload reduced 70%",
  },
  {
    icon: TrendingUp,
    metric: "3x",
    metricLabel: "Lead Response Velocity",
    title: "Sales Agent for Janjua Global",
    description: "Autonomous AI sales bot integration",
    detail: "Replaced manual follow-ups with an AI agent that handles queries 24/7 with human-level accuracy.",
  },
  {
    icon: BarChart3,
    metric: "98%",
    metricLabel: "Data Extraction Accuracy",
    title: "DocuSense AI for Daak Khana",
    description: "Vision-AI processing platform",
    detail: "Automated extraction from 10,000+ hand-written logs monthly using custom fine-tuned Vision LLMs.",
  },
  {
    icon: Clock,
    metric: "80%",
    metricLabel: "Parent Communication",
    title: "SchoolSync for APS Jinnah",
    description: "School administration system",
    detail: "800+ students and staff with automated reporting and parent portal",
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

        {/* SaaS Case Study Detail */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] as const }}
          className="mt-16 sm:mt-20"
        >
          <div className="p-8 sm:p-12 rounded-3xl bg-white/[0.02] border border-white/10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 mb-4">
                  <span className="text-xs text-white/60">AI Systems Case Study</span>
                </div>
                <h3 className="text-2xl sm:text-3xl font-light text-white mb-4">
                  Autonomous Marketing Engine
                </h3>
                <p className="text-sm sm:text-base text-white/60 mb-6 leading-relaxed">
                  A high-growth startup needed to scale content and outreach without hiring a massive team. We built a custom multi-agent system that researches, writes, and distributes content autonomously.
                </p>
                <div className="flex flex-wrap gap-6 mb-6">
                  <div>
                    <p className="text-3xl font-light text-white">14</p>
                    <p className="text-xs text-white/40">Days to Deploy</p>
                  </div>
                  <div>
                    <p className="text-3xl font-light text-white">400%</p>
                    <p className="text-xs text-white/40">Content Volume Increase</p>
                  </div>
                  <div>
                    <p className="text-3xl font-light text-white">$15k+</p>
                    <p className="text-xs text-white/40">Project Value</p>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <div className="flex items-start gap-3">
                    <span className="text-white/40 text-sm">Problem:</span>
                    <p className="text-sm text-white/60">Manual reporting taking 40 hours/week, customers churning due to lack of insights</p>
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <div className="flex items-start gap-3">
                    <span className="text-white/40 text-sm">Solution:</span>
                    <p className="text-sm text-white/60">Automated real-time analytics dashboard with predictive forecasting</p>
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <div className="flex items-start gap-3">
                    <span className="text-white/40 text-sm">Tech Stack:</span>
                    <p className="text-sm text-white/60">React, Node.js, PostgreSQL, AWS, TensorFlow for predictions</p>
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/10">
                  <div className="flex items-start gap-3">
                    <span className="text-white/40 text-sm">Result:</span>
                    <p className="text-sm text-white/60">Reporting time reduced to 2 hours/week, customer retention up 35%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
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
            <span className="text-white/80">Built 3 startups</span> and learned what actually matters. 
            <span className="text-white/80"> Trusted by 50+ founders</span> to deliver systems that drive real business growth.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
