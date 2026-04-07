"use client";

import { motion } from "framer-motion";
import { Users } from "lucide-react";

// Cartoonish Avatar Component with gradient backgrounds
const CartoonAvatar = ({ name, className }: { name: string; className?: string }) => {
  // Generate consistent color based on name
  const colors = [
    { bg: "from-blue-500/30 to-blue-600/20", accent: "bg-blue-400" },
    { bg: "from-emerald-500/30 to-emerald-600/20", accent: "bg-emerald-400" },
    { bg: "from-purple-500/30 to-purple-600/20", accent: "bg-purple-400" },
    { bg: "from-amber-500/30 to-amber-600/20", accent: "bg-amber-400" },
    { bg: "from-rose-500/30 to-rose-600/20", accent: "bg-rose-400" },
    { bg: "from-cyan-500/30 to-cyan-600/20", accent: "bg-cyan-400" },
  ];
  
  const getColorIndex = (str: string) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return Math.abs(hash) % colors.length;
  };
  
  const color = colors[getColorIndex(name)];
  
  return (
    <div className={`relative rounded-full overflow-hidden bg-gradient-to-br ${color.bg} ${className}`}>
      {/* Cartoon face SVG */}
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {/* Face base */}
        <circle cx="50" cy="45" r="35" fill="rgba(255,255,255,0.15)" />
        {/* Eyes */}
        <circle cx="38" cy="40" r="5" fill="rgba(255,255,255,0.8)" />
        <circle cx="62" cy="40" r="5" fill="rgba(255,255,255,0.8)" />
        {/* Smile */}
        <path d="M 35 55 Q 50 65 65 55" stroke="rgba(255,255,255,0.6)" strokeWidth="2" fill="none" strokeLinecap="round" />
        {/* Decorative elements */}
        <circle cx="20" cy="25" r="3" fill="rgba(255,255,255,0.2)" />
        <circle cx="80" cy="30" r="2" fill="rgba(255,255,255,0.2)" />
        <circle cx="75" cy="70" r="4" fill="rgba(255,255,255,0.15)" />
      </svg>
    </div>
  );
};

// Custom LinkedIn icon
const LinkedinIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5 text-white/60">
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

const team = [
  {
    name: "Ahmad Jamil",
    role: "Founder & CEO",
    bio: "Visionary leader driving innovation and growth at zehanx Technologies with expertise in business strategy and technology.",
    social: {
      linkedin: "#",
    },
  },
  {
    name: "Ahmad Ibrahim",
    role: "Co-founder & COO",
    bio: "Operations expert ensuring seamless delivery of projects and maintaining excellence in client relationships.",
    social: {
      linkedin: "#",
    },
  },
  {
    name: "Umair Amin",
    role: "Co-founder & CMO",
    bio: "Marketing strategist expanding zehanx reach globally and building strong brand presence in the tech industry.",
    social: {
      linkedin: "#",
    },
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

export default function Team() {
  return (
    <section id="team" className="relative w-full bg-black py-20 sm:py-24 lg:py-32 overflow-hidden">
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
            <Users className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Our Team</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Meet the Experts
            <br className="hidden sm:block" />
            Behind zehanx
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            A passionate team of developers, designers, and AI specialists dedicated to delivering excellence.
          </p>
        </motion.div>

        {/* Team Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 max-w-4xl mx-auto"
        >
          {team.map((member) => (
            <motion.div
              key={member.name}
              variants={itemVariants}
              className="group relative"
            >
              <div className="relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300 text-center">
                {/* Profile Avatar - Cartoon Style */}
                <div className="relative w-24 h-24 sm:w-32 sm:h-32 mx-auto mb-4 rounded-full overflow-hidden ring-2 ring-white/10 ring-offset-2 ring-offset-black">
                  <CartoonAvatar name={member.name} className="w-full h-full" />
                </div>

                {/* Info */}
                <h3 className="text-lg sm:text-xl font-light text-white mb-1">
                  {member.name}
                </h3>
                <p className="text-sm text-white/50 mb-3">
                  {member.role}
                </p>
                <p className="text-xs sm:text-sm text-white/40 mb-4 leading-relaxed">
                  {member.bio}
                </p>

                {/* Social Links */}
                <div className="flex items-center justify-center">
                  <a
                    href={member.social.linkedin}
                    className="flex items-center justify-center w-8 h-8 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all"
                    aria-label={`${member.name} LinkedIn`}
                  >
                    <LinkedinIcon />
                  </a>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
