"use client";

import { motion } from "framer-motion";
import { Check, ArrowRight } from "lucide-react";

const plans = [
  {
    name: "Starter",
    price: "Custom",
    description: "Perfect for startups and small projects",
    features: [
      "Web Development (5 pages)",
      "Basic AI Integration",
      "Mobile Responsive Design",
      "3 Months Support",
      "Source Code Delivery",
    ],
    cta: "Get Started",
    popular: false,
  },
  {
    name: "Professional",
    price: "Custom",
    description: "Ideal for growing businesses",
    features: [
      "Full-Stack Web Application",
      "Advanced ML/AI Solutions",
      "iOS & Android Apps",
      "API Development",
      "6 Months Support",
      "Performance Optimization",
    ],
    cta: "Get Started",
    popular: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For large-scale enterprise solutions",
    features: [
      "Custom Enterprise Software",
      "Deep Learning & Neural Networks",
      "Multi-platform Solutions",
      "Cloud Infrastructure",
      "12 Months Priority Support",
      "Dedicated Project Manager",
      "24/7 Maintenance",
    ],
    cta: "Contact Us",
    popular: false,
  },
];

export default function Pricing() {
  return (
    <section className="relative w-full bg-black py-20 sm:py-24 lg:py-32">
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
            <span className="text-sm text-white/80">Pricing</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-light text-white tracking-tight mb-4 sm:mb-6">
            Flexible Solutions
            <br className="hidden sm:block" />
            for Every Need
          </h2>
          <p className="text-base sm:text-lg text-white/60 max-w-2xl mx-auto px-4 sm:px-0">
            Custom pricing tailored to your project requirements. Contact us for a detailed quote.
          </p>
        </motion.div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{
                duration: 0.6,
                delay: index * 0.1,
                ease: [0.16, 1, 0.3, 1] as const,
              }}
              className={`relative p-6 sm:p-8 rounded-2xl border ${
                plan.popular
                  ? "bg-white/[0.04] border-white/20"
                  : "bg-white/[0.02] border-white/10"
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="px-3 py-1 text-xs font-light text-black bg-white rounded-full">
                    Most Popular
                  </span>
                </div>
              )}

              <div className="mb-6">
                <h3 className="text-xl font-light text-white mb-2">{plan.name}</h3>
                <p className="text-sm text-white/50 mb-4">{plan.description}</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl sm:text-4xl font-light text-white">{plan.price}</span>
                  <span className="text-white/50">Quote</span>
                </div>
              </div>

              <ul className="space-y-3 mb-8">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3">
                    <Check className="w-5 h-5 text-white/60 shrink-0 mt-0.5" />
                    <span className="text-sm text-white/70">{feature}</span>
                  </li>
                ))}
              </ul>

              <a
                href="#contact"
                className={`flex items-center justify-center gap-2 w-full py-3 rounded-full font-light transition-all ${
                  plan.popular
                    ? "text-black bg-gradient-to-b from-white to-gray-200 hover:from-gray-100 hover:to-gray-300"
                    : "text-white bg-white/5 border border-white/10 hover:bg-white/10"
                }`}
              >
                {plan.cta}
                <ArrowRight className="w-4 h-4" />
              </a>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
