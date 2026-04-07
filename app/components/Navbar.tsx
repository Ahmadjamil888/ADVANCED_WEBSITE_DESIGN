"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import NextImage from "next/image";

interface NavLinkProps {
  href: string;
  children: React.ReactNode;
  isActive?: boolean;
  isStrikethrough?: boolean;
  onClick?: () => void;
}

function NavLink({ href, children, isActive, isStrikethrough, onClick }: NavLinkProps) {
  return (
    <a
      href={href}
      onClick={onClick}
      className={cn(
        "text-sm font-medium text-white/80 hover:text-white transition-colors relative px-4 py-2 block w-full sm:w-auto",
        isActive && "text-white",
        isStrikethrough && "line-through opacity-50"
      )}
    >
      {isActive && (
        <span className="absolute inset-0 rounded-full border border-transparent bg-gradient-to-r from-white/20 via-white/40 to-white/20 p-[1px]">
          <span className="block h-full w-full rounded-full bg-black/50 backdrop-blur-sm" />
        </span>
      )}
      <span className="relative z-10">{children}</span>
    </a>
  );
}

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] as const }}
      className="fixed top-0 left-0 right-0 z-50 h-16"
    >
      <div className="absolute inset-0 bg-black/20 backdrop-blur-xl border-b border-white/5" />
      <div className="relative h-full max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="relative h-8 w-auto">
          <NextImage 
            src="/logo.png" 
            alt="zehanx Technologies" 
            width={120} 
            height={32} 
            className="h-8 w-auto object-contain"
            priority
          />
        </a>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-1">
          <NavLink href="/features" isActive>
            Features
          </NavLink>
          <NavLink href="/insights">Insights</NavLink>
          <NavLink href="/about">About</NavLink>
          <NavLink href="/services">Services</NavLink>
          <NavLink href="/contact">Contact</NavLink>
        </div>

        {/* Desktop CTA */}
        <a
          href="https://cal.com/zehanx-technologies-official"
          className="hidden md:block px-4 sm:px-5 py-2 sm:py-2.5 rounded-full text-sm font-medium text-black bg-gradient-to-b from-white to-gray-200 hover:from-gray-100 hover:to-gray-300 transition-all shadow-lg shadow-white/10"
        >
          Get Started
        </a>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="md:hidden p-2 text-white/80 hover:text-white transition-colors"
          aria-label="Toggle menu"
        >
          {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden absolute top-full left-0 right-0 bg-black/95 backdrop-blur-xl border-b border-white/10"
          >
            <div className="px-4 py-6 flex flex-col gap-2">
              <NavLink href="/features" isActive onClick={() => setIsOpen(false)}>
                Features
              </NavLink>
              <NavLink href="/insights" onClick={() => setIsOpen(false)}>
                Insights
              </NavLink>
              <NavLink href="/about" onClick={() => setIsOpen(false)}>
                About
              </NavLink>
              <NavLink href="/services" onClick={() => setIsOpen(false)}>
                Services
              </NavLink>
              <NavLink href="/contact" onClick={() => setIsOpen(false)}>
                Contact
              </NavLink>
              <a
                href="https://cal.com/zehanx-technologies-official"
                onClick={() => setIsOpen(false)}
                className="mt-4 w-full text-center px-5 py-3 rounded-full text-sm font-medium text-black bg-gradient-to-b from-white to-gray-200 hover:from-gray-100 hover:to-gray-300 transition-all"
              >
                Book a Call
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
