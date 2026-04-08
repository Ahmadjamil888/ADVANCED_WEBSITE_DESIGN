"use client";

import { motion } from "framer-motion";
import { Building2, GraduationCap, Stethoscope, Truck, Scissors } from "lucide-react";

const clients = [
  {
    name: "Dr. Saira",
    role: "CEO, Usman Hospital",
    industry: "Healthcare",
    project: "BrainSkills AI",
    icon: Stethoscope,
  },
  {
    name: "Rana Asif Khan",
    role: "CEO, IRTCoP",
    industry: "Education",
    project: "Institute Management",
    icon: GraduationCap,
  },
  {
    name: "Umair Fiaz",
    role: "CEO, Janjua Tailors",
    industry: "Retail/SMB",
    project: "OrderFlow Pro",
    icon: Scissors,
  },
  {
    name: "Syeda Eyesha Nadeem",
    role: "CEO, APS Jinnah",
    industry: "Education",
    project: "SchoolSync",
    icon: Building2,
  },
  {
    name: "Shazab Jamil",
    role: "CEO, Daak Khana",
    industry: "Logistics",
    project: "LogiTrack",
    icon: Truck,
  },
];

export default function TrustedBy() {
  return (
    <section className="relative w-full bg-black py-16 sm:py-20 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-white/[0.01] to-black pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-10 sm:mb-12"
        >
          <p className="text-sm text-white/40 tracking-wider uppercase mb-2">
            Trusted by Founders & Business Owners
          </p>
          <p className="text-xs sm:text-sm text-white/30 max-w-xl mx-auto">
            Across hospitals, schools, logistics, and retail — we build systems real businesses depend on
          </p>
        </motion.div>

        {/* Client Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] as const }}
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6"
        >
          {clients.map((client, index) => (
            <motion.div
              key={client.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group relative p-4 sm:p-6 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
            >
              {/* Icon */}
              <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-3">
                <client.icon className="w-5 h-5 text-white/50" />
              </div>

              {/* Content */}
              <p className="text-sm font-light text-white mb-1">{client.name}</p>
              <p className="text-xs text-white/40 mb-2">{client.role}</p>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-white/30 uppercase tracking-wider">{client.industry}</span>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Stats Row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-12 flex flex-wrap items-center justify-center gap-8 sm:gap-12"
        >
          <div className="text-center">
            <p className="text-2xl sm:text-3xl font-light text-white">50+</p>
            <p className="text-xs text-white/40">Systems Shipped</p>
          </div>
          <div className="w-px h-8 bg-white/10 hidden sm:block" />
          <div className="text-center">
            <p className="text-2xl sm:text-3xl font-light text-white">21</p>
            <p className="text-xs text-white/40">Days Average</p>
          </div>
          <div className="w-px h-8 bg-white/10 hidden sm:block" />
          <div className="text-center">
            <p className="text-2xl sm:text-3xl font-light text-white">5</p>
            <p className="text-xs text-white/40">Industries Served</p>
          </div>
          <div className="w-px h-8 bg-white/10 hidden sm:block" />
          <div className="text-center">
            <p className="text-2xl sm:text-3xl font-light text-white">$10k–$50k</p>
            <p className="text-xs text-white/40">Project Range</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
