"use client";

import { motion } from "framer-motion";
import { Quote, Star } from "lucide-react";

const testimonials = [
  {
    quote: "zehanx Technologies transformed our business with their AI-powered solution. The team's expertise in machine learning is unmatched.",
    author: "Sarah Johnson",
    role: "CTO, TechStart Inc.",
    rating: 5,
    image: "/avatar-2.png",
  },
  {
    quote: "The web development team delivered a stunning, high-performance website that exceeded our expectations. Highly recommended!",
    author: "Michael Chen",
    role: "Founder, GrowthLabs",
    rating: 5,
    image: "/avatar-5.png",
  },
  {
    quote: "Working with zehanx on our mobile app was a game-changer. Their attention to detail and technical expertise is world-class.",
    author: "David Rodriguez",
    role: "Data Director, FinanceHub",
    rating: 5,
    image: "/avatar-6.png",
  },
  {
    quote: "Their deep learning solution helped us automate complex processes, saving us thousands of hours annually. Exceptional work!",
    author: "Fatima Rizvi",
    role: "Director, DataSystems",
    rating: 5,
    image: "/avatar-4.png",
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
            Trusted by Industry
            <br className="hidden sm:block" />
            Leaders
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Hear from our clients about their experience working with zehanx Technologies.
          </p>
        </motion.div>

        {/* Testimonials Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8"
        >
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className="relative p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10"
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
