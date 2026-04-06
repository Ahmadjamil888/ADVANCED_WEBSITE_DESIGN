"use client";

import { motion } from "framer-motion";
import { Target, Lightbulb, Users, Award, Eye, Compass, Sparkles } from "lucide-react";

const values = [
  {
    icon: Target,
    title: "6+ Years Experience",
    description: "Partnered and served clients for more than 6 years with consistent quality and dedication.",
  },
  {
    icon: Lightbulb,
    title: "AI & ML Expertise",
    description: "Deep expertise in Artificial Intelligence, Machine Learning, Deep Learning, and Neural Networks.",
  },
  {
    icon: Users,
    title: "Full-Stack Solutions",
    description: "Complete web, software, and app development services from concept to deployment.",
  },
  {
    icon: Award,
    title: "Trusted Partnerships",
    description: "Long-term relationships with clients built on trust, quality, and exceptional results.",
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
              <span className="text-sm text-white/80">About Us</span>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-semibold text-white tracking-tight mb-4 sm:mb-6">
              Delivering excellence
              <span className="block text-white/60">for 6+ years</span>
            </h2>
            <p className="text-base sm:text-lg text-white/60 leading-relaxed mb-6 sm:mb-8">
              zehanx Technologies has been at the forefront of innovation, providing comprehensive technology solutions. From web development to advanced AI systems, we have partnered with businesses worldwide for over 6 years.
            </p>
            <p className="text-base sm:text-lg text-white/60 leading-relaxed">
              Our expertise spans Web Development, AI, Machine Learning, Deep Learning, Neural Networks, Software Development, and App Development. We transform ideas into reality.
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
              To empower businesses worldwide with cutting-edge technology solutions that drive growth, efficiency, and innovation. We strive to bridge the gap between complex technology and practical business applications.
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
              To be a global leader in AI and technology solutions, recognized for innovation, quality, and transformative impact. We envision a future where every business harnesses the power of advanced technology.
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
