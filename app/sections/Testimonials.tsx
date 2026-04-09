"use client";

import { motion } from "framer-motion";
import { Quote, Star } from "lucide-react";

const testimonials = [
  {
    quote: "We needed a system to manage hospital operations digitally. Zehanx delivered a complete solution that improved efficiency and reduced manual work significantly.",
    author: "Dr. Saira",
    role: "CEO, Usman Hospital",
    rating: 5,
    image: "/avatar-1.png",
  },
  {
    quote: "Professional team. Delivered exactly what we needed with clarity and strong technical execution. Our institute management system now handles 500+ students seamlessly.",
    author: "Rana Asif Khan",
    role: "CEO, IRTCoP",
    rating: 5,
    image: "/avatar-9.png",
  },
  {
    quote: "They digitized our workflow and made our operations much more efficient. What used to take days now happens in hours. The system paid for itself in two months.",
    author: "Umair Fiaz",
    role: "CEO, Janjua Tailors",
    rating: 5,
    image: "/avatar-3.png",
  },
  {
    quote: "They built a custom platform that streamlined our school operations. What stood out was how fast they shipped — and how well they understood our needs as an institution.",
    author: "Syeda Eyesha Nadeem",
    role: "CEO, APS Jinnah",
    rating: 5,
    image: "/avatar-8.png",
  },
  {
    quote: "Zehanx built our logistics tracking system in under a month. Now we handle 10,000+ packages monthly with 98% delivery accuracy. Their speed and execution were impressive.",
    author: "Shazab Jamil",
    role: "CEO, Daak Khana",
    rating: 5,
    image: "/avatar-2.png",
  },
  {
    quote: "Zehanx Technologies built our PSX ledger and now it has over 900k users and a handful cash flow, I highly recommend purchasing their saas development plan.",
    author: "Zain shafeeq",
    role: "CEO, mrowldesign",
    rating: 5,
    image: "/avatar-7.png",
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
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

export default function Testimonials() {
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
            <Quote className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Testimonials</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Trusted by Founders
            <br className="hidden sm:block" />
            & Business Owners
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Across hospitals, schools, logistics, and retail — we build systems that real businesses depend on.
          </p>
        </motion.div>

        {/* Testimonials Grid - 5 testimonials */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8"
        >
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className={`relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10 ${index === 4 ? 'md:col-span-2 lg:col-span-1' : ''}`}
            >
              {/* Quote icon */}
              <div className="absolute top-6 right-6 sm:top-8 sm:right-8">
                <Quote className="w-8 h-8 text-white/10" />
              </div>

              {/* Stars */}
              <div className="flex gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 fill-white/80 text-white/80" />
                ))}
              </div>

              {/* Quote text */}
              <p className="text-base sm:text-lg text-white/80 leading-relaxed mb-6 font-light">
                &ldquo;{testimonial.quote}&rdquo;
              </p>

              {/* Author */}
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full overflow-hidden bg-white/10">
                  <img
                    src={testimonial.image}
                    alt={testimonial.author}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div>
                  <h4 className="text-white font-light">{testimonial.author}</h4>
                  <p className="text-sm text-white/50">{testimonial.role}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
