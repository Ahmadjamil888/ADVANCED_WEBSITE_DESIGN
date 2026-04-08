"use client";

import { motion } from "framer-motion";
import { Target, Lightbulb, Users, Award, Eye, Compass, Sparkles } from "lucide-react";

const values = [
  {
    icon: Target,
    title: "Built 3 Startups",
    description: "Failed, learned, succeeded. We have lived the founder journey and understand what actually matters.",
  },
  {
    icon: Lightbulb,
    title: "AI-First Engineering",
    description: "Not just AI tools — we architect systems where AI is the core competitive advantage.",
  },
  {
    icon: Users,
    title: "Speed to Market",
    description: "21-day MVP system. Get to revenue faster without sacrificing quality or scalability.",
  },
  {
    icon: Award,
    title: "Product Partnership",
    description: "We do not just write code. We think product, business, and user experience with you.",
  },
];

const process = [
  {
    step: "01",
    title: "Discovery",
    description: "We analyze your requirements, understand your goals, and define the project scope.",
  },
  {
    step: "02",
    title: "Strategy",
    description: "Our team crafts a detailed roadmap with timelines, milestones, and deliverables.",
  },
  {
    step: "03",
    title: "Development",
    description: "We build your solution using agile methodologies with regular updates and feedback.",
  },
  {
    step: "04",
    title: "Deployment",
    description: "After rigorous testing, we deploy your solution and provide ongoing support.",
  },
];

export default function About() {
  return (
    <section id="about" className="relative w-full bg-black py-20 sm:py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main About Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-center mb-20 sm:mb-32">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
              <Award className="w-4 h-4 text-white/70" />
              <span className="text-sm text-white/80">Founder Story</span>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-white tracking-tight mb-4 sm:mb-6">
              We Have Been
              <span className="block text-white/60">Where You Are</span>
            </h2>
            <p className="text-base sm:text-lg text-white/60 leading-relaxed mb-6 sm:mb-8">
              Built 3 startups. Faced the chaos of early-stage funding, technical debt, and the pressure to ship. We spent months building what should have taken weeks.
            </p>
            <p className="text-base sm:text-lg text-white/60 leading-relaxed">
              That is why we created Zehanx — to be the technical partner we wish we had. Someone who moves fast, thinks business-first, and actually ships production-ready systems.
            </p>
          </motion.div>

          {/* Right Content - Values */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ 
                  duration: 0.6, 
                  delay: index * 0.1,
                  ease: [0.16, 1, 0.3, 1] as const 
                }}
                className="p-5 sm:p-6 rounded-xl bg-white/[0.02] border border-white/10"
              >
                <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-white/5 mb-3 sm:mb-4">
                  <value.icon className="w-5 h-5 sm:w-6 sm:h-6 text-white/80" />
                </div>
                <h3 className="text-base sm:text-lg font-semibold text-white mb-2">
                  {value.title}
                </h3>
                <p className="text-sm text-white/60 leading-relaxed">
                  {value.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mission & Vision Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 mb-20 sm:mb-32">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
            className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/5">
                <Compass className="w-5 h-5 text-white/70" />
              </div>
              <h3 className="text-xl sm:text-2xl font-light text-white">Our Mission</h3>
            </div>
            <p className="text-sm sm:text-base text-white/60 leading-relaxed">
              Help SaaS founders escape the build trap. Ship revenue-ready systems in weeks, not quarters. Be the technical co-founder you do not have to hire full-time.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] as const }}
            className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/5">
                <Eye className="w-5 h-5 text-white/70" />
              </div>
              <h3 className="text-xl sm:text-2xl font-light text-white">Our Vision</h3>
            </div>
            <p className="text-sm sm:text-base text-white/60 leading-relaxed">
              Every founder deserves a world-class technical team from day one. We are building that reality — one AI-powered system at a time.
            </p>
          </motion.div>
        </div>

        {/* Process Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Sparkles className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Our Process</span>
          </div>
          <h3 className="text-2xl sm:text-3xl lg:text-4xl font-light text-white tracking-tight mb-4">
            How we work
          </h3>
          <p className="text-base sm:text-lg text-white/60 max-w-xl mx-auto">
            A streamlined approach to delivering exceptional results
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          {process.map((item, index) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                duration: 0.6, 
                delay: index * 0.1,
                ease: [0.16, 1, 0.3, 1] as const 
              }}
              className="relative p-5 sm:p-6 rounded-xl bg-white/[0.02] border border-white/10"
            >
              <span className="absolute top-4 right-4 text-3xl sm:text-4xl font-bold text-white/5">
                {item.step}
              </span>
              <h4 className="text-lg font-medium text-white mb-2">{item.title}</h4>
              <p className="text-sm text-white/60 leading-relaxed">{item.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
