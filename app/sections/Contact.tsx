"use client";

import { motion } from "framer-motion";
import { Mail, MapPin, Phone, MessageCircle, Calendar, ArrowUpRight } from "lucide-react";

const contactInfo = [
  {
    icon: Mail,
    label: "Email",
    value: "zehanxtech@gmail.com",
    href: "mailto:zehanxtech@gmail.com",
  },
  {
    icon: MessageCircle,
    label: "WhatsApp",
    value: "+92 333 8188722",
    href: "https://wa.me/923338188722",
  },
  {
    icon: Phone,
    label: "Phone",
    value: "+92 333 8188722",
    href: "tel:+923338188722",
  },
  {
    icon: MapPin,
    label: "Location",
    value: "Pakistan",
    href: "#",
  },
];

export default function Contact() {
  return (
    <section id="contact" className="relative w-full bg-black py-20 sm:py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
          className="text-center mb-12 sm:mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10 mb-6">
            <Mail className="w-4 h-4 text-white/70" />
            <span className="text-sm text-white/80">Contact</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-semibold text-white tracking-tight mb-4 sm:mb-6">
            Let&apos;s build your
            <br className="hidden sm:block" />
            next big thing
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Ready to transform your ideas into reality? Contact zehanx Technologies for Web, AI, ML, and App solutions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 max-w-5xl mx-auto">
          {/* Contact Info */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
            className="space-y-4"
          >
            <h3 className="text-xl sm:text-2xl font-light text-white mb-6">Get in Touch</h3>
            {contactInfo.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="flex items-center gap-4 p-4 sm:p-5 rounded-xl bg-white/[0.02] border border-white/10 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300"
              >
                <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-white/5 shrink-0">
                  <item.icon className="w-5 h-5 sm:w-6 sm:h-6 text-white/80" />
                </div>
                <div>
                  <h4 className="text-sm text-white/50 mb-1">{item.label}</h4>
                  <p className="text-base sm:text-lg font-medium text-white">{item.value}</p>
                </div>
              </a>
            ))}
          </motion.div>

          {/* Booking CTA */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] as const }}
            className="flex flex-col justify-center"
          >
            <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/10">
              <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-white/5 mb-6">
                <Calendar className="w-7 h-7 text-white/80" />
              </div>
              <h3 className="text-xl sm:text-2xl font-light text-white mb-3">
                Book a Meeting
              </h3>
              <p className="text-sm sm:text-base text-white/60 leading-relaxed mb-6">
                Schedule a free consultation call with our team to discuss your project requirements and explore how we can help transform your business.
              </p>
              <a
                href="https://cal.com/zehanx-technologies-official"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 sm:px-8 py-3 sm:py-4 rounded-full text-black font-medium bg-gradient-to-b from-white to-gray-200 hover:from-gray-100 hover:to-gray-300 transition-all shadow-lg shadow-white/10"
              >
                Schedule a Call
                <ArrowUpRight className="w-4 h-4" />
              </a>
              <p className="text-xs text-white/40 mt-4">
                Powered by Cal.com · Free 30-minute consultation
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
