"use client";

import { motion } from "framer-motion";
import { 
  Zap, 
  Shield, 
  Rocket, 
  BarChart3, 
  Globe, 
  Lock,
  ArrowRight,
  Check,
  Cpu,
  Code2,
  Database,
  Cloud,
  Layers
} from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "Custom LLM Infrastructure",
    description: "End-to-end development of custom-trained and fine-tuned LLMs tailored to your proprietary datasets.",
  },
  {
    icon: Zap,
    title: "Autonomous AI Agents",
    description: "Self-correcting AI agents that handle sales, support, and internal operations without human intervention.",
  },
  {
    icon: Database,
    title: "Enterprise RAG Systems",
    description: "Production-ready Vector DB architectures that allow your business to chat with its data in real-time.",
  },
  {
    icon: BarChart3,
    title: "Workflow Automation",
    description: "Replacing manual business processes with 24/7 autonomous loops that save 20+ hours per week.",
  },
  {
    icon: Shield,
    title: "AI Security & Guardrails",
    description: "Enterprise-grade safety layers, data privacy controls, and prompt injection protection for your AI assets.",
  },
  {
    icon: Rocket,
    title: "High-Velocity Deployment",
    description: "We ship revenue-generating AI systems in 7–21 days using our high-velocity engineering framework.",
  },
];

const techStack = [
  { icon: Code2, name: "React & Next.js", category: "Frontend" },
  { icon: Database, name: "Python & TensorFlow", category: "AI/ML" },
  { icon: Cloud, name: "AWS & Cloud", category: "Infrastructure" },
  { icon: Layers, name: "Node.js & Express", category: "Backend" },
  { icon: Cpu, name: "PyTorch & Keras", category: "Deep Learning" },
  { icon: Globe, name: "React Native", category: "Mobile" },
];

const whyChooseUs = [
  "6+ years of industry experience",
  "Expert team of AI/ML specialists",
  "End-to-end development services",
  "Agile development methodology",
  "24/7 technical support",
  "On-time project delivery",
  "Transparent communication",
  "Post-launch maintenance",
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

export default function Features() {
  return (
    <section id="features" className="relative w-full bg-black py-20 sm:py-24 lg:py-32">
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
            <Zap className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Features</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-semibold text-white tracking-tight mb-4 sm:mb-6">
            Everything you need to
            <br className="hidden sm:block" />
            automate your world
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            From custom LLMs to autonomous agent networks, we build the technical infrastructure that enables exponential growth.
          </p>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 mb-20 sm:mb-32"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              className="group relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-white/5 mb-4 sm:mb-6">
                <feature.icon className="w-6 h-6 text-white/80" />
              </div>
              <h3 className="text-lg sm:text-xl font-semibold text-white mb-2 sm:mb-3">
                {feature.title}
              </h3>
              <p className="text-sm sm:text-base text-white/60 leading-relaxed">
                {feature.description}
              </p>
              <div className="mt-4 sm:mt-6 flex items-center gap-2 text-white/40 group-hover:text-white/80 transition-colors">
                <span className="text-sm font-medium">Learn more</span>
                <ArrowRight className="w-4 h-4" />
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Technology Stack Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Cpu className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Built for Results</span>
          </div>
          <h3 className="text-2xl sm:text-3xl lg:text-4xl font-light text-white tracking-tight mb-4">
            Built for speed, scale, and real-world impact
          </h3>
          <p className="text-base sm:text-lg text-white/60 max-w-xl mx-auto">
            We leverage proven technologies that deliver results — not just code that looks good on paper.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 sm:gap-6 mb-20 sm:mb-32"
        >
          {techStack.map((tech) => (
            <motion.div
              key={tech.name}
              variants={itemVariants}
              className="group relative p-4 sm:p-6 rounded-xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300 text-center"
            >
              <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-white/5 mx-auto mb-3">
                <tech.icon className="w-5 h-5 sm:w-6 sm:h-6 text-white/70" />
              </div>
              <p className="text-xs text-white/40 mb-1">{tech.category}</p>
              <p className="text-sm font-medium text-white">{tech.name}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Why Choose Us Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
              <Check className="w-4 h-4 text-white/70" />
              <span className="text-sm text-white/80">Why Choose Us</span>
            </div>
            <h3 className="text-2xl sm:text-3xl lg:text-4xl font-light text-white tracking-tight mb-4">
              Trusted by businesses worldwide
            </h3>
            <p className="text-base sm:text-lg text-white/60 leading-relaxed mb-6">
              We combine technical expertise with business acumen to deliver solutions that drive real results for your organization.
            </p>
            <a
              href="https://cal.com/zehanx-technologies-official"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-full text-black font-medium bg-gradient-to-b from-white to-gray-200 hover:from-gray-100 hover:to-gray-300 transition-all shadow-lg shadow-white/10"
            >
              Start Your AI Project
              <ArrowRight className="w-4 h-4" />
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] as const }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-3"
          >
            {whyChooseUs.map((item, index) => (
              <motion.div
                key={item}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.05 }}
                className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.02] border border-white/10"
              >
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-white/10 shrink-0">
                  <Check className="w-3.5 h-3.5 text-white/70" />
                </div>
                <span className="text-sm text-white/80">{item}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
