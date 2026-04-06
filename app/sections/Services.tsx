"use client";

import { motion } from "framer-motion";
import { Code, Brain, Bot, Network, Smartphone, Globe, ArrowRight } from "lucide-react";

const services = [
  {
    icon: Code,
    title: "Web Development",
    description: "Custom websites, web applications, and e-commerce solutions built with modern frameworks like React, Next.js, and Node.js.",
    tags: ["React", "Next.js", "Node.js", "Full-Stack"],
  },
  {
    icon: Brain,
    title: "Artificial Intelligence",
    description: "AI-powered solutions for automation, predictive analytics, natural language processing, and computer vision applications.",
    tags: ["NLP", "Computer Vision", "Predictive Analytics"],
  },
  {
    icon: Bot,
    title: "Machine Learning",
    description: "End-to-end ML solutions from data preprocessing to model deployment, including supervised and unsupervised learning systems.",
    tags: ["TensorFlow", "PyTorch", "Scikit-learn"],
  },
  {
    icon: Network,
    title: "Deep Learning & Neural Networks",
    description: "Advanced neural network architectures for complex pattern recognition, image processing, and deep learning applications.",
    tags: ["CNN", "RNN", "Transformers", "GANs"],
  },
  {
    icon: Smartphone,
    title: "App Development",
    description: "Native and cross-platform mobile applications for iOS and Android with seamless user experiences and robust backend integration.",
    tags: ["iOS", "Android", "React Native", "Flutter"],
  },
  {
    icon: Globe,
    title: "Software Development",
    description: "Custom enterprise software, desktop applications, and system integrations tailored to your business requirements.",
    tags: ["Enterprise", "Desktop", "APIs", "Cloud"],
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
            <span className="text-sm text-white/80">Our Services</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Comprehensive Technology
            <br className="hidden sm:block" />
            Solutions
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            From concept to deployment, we deliver cutting-edge solutions that drive business growth and innovation.
          </p>
        </motion.div>

        {/* Services Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8"
        >
          {services.map((service) => (
            <motion.div
              key={service.title}
              variants={itemVariants}
              className="group relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-white/5 mb-4 sm:mb-6">
                <service.icon className="w-6 h-6 text-white/80" />
              </div>
              <h3 className="text-lg sm:text-xl font-light text-white mb-2 sm:mb-3">
                {service.title}
              </h3>
              <p className="text-sm sm:text-base text-white/60 leading-relaxed mb-4">
                {service.description}
              </p>
              <div className="flex flex-wrap gap-2 mb-4">
                {service.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 text-xs text-white/50 bg-white/5 rounded-full border border-white/10"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <div className="flex items-center gap-2 text-white/40 group-hover:text-white/80 transition-colors">
                <span className="text-sm font-light">Learn more</span>
                <ArrowRight className="w-4 h-4" />
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
