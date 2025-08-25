'use client';

import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavbarLink {
  text: string;
  href: string;
}

interface ModernNavbarProps {
  name?: string;
  homeUrl?: string;
  links?: NavbarLink[];
  className?: string;
}

export default function ModernNavbar({
  name = 'ZEHANX',
  homeUrl = '/',
  links = [
    { text: 'Home', href: '/' },
    { text: 'Services', href: '/services' },
    { text: 'Portfolio', href: '/portfolio' },
    { text: 'About', href: '/about' },
    { text: 'Blog', href: '/blog' },
  ],
  className,
}: ModernNavbarProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <>
      <nav className={cn('fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm', className)}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <a href={homeUrl} className="flex items-center">
              <span className="text-2xl font-bold text-white">{name}</span>
            </a>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              {links.map((link, index) => (
                <a
                  key={index}
                  href={link.href}
                  className="text-white hover:text-gray-300 transition-colors duration-200 font-medium"
                >
                  {link.text}
                </a>
              ))}
            </div>

            {/* CTA Button */}
            <div className="hidden md:block">
              <a
                href="/contact"
                className="bg-white text-black px-6 py-2 rounded-lg font-semibold hover:bg-gray-200 transition-colors duration-200"
              >
                Get In Touch
              </a>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={toggleMobileMenu}
              className="md:hidden text-white"
            >
              {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/90" />
          <div className="relative z-50 flex flex-col items-center justify-center h-full space-y-8">
            {links.map((link, index) => (
              <a
                key={index}
                href={link.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className="text-2xl text-white font-medium"
              >
                {link.text}
              </a>
            ))}
            <a
              href="/contact"
              onClick={() => setIsMobileMenuOpen(false)}
              className="mt-8 bg-white text-black px-8 py-3 rounded-lg font-semibold text-lg"
            >
              Get In Touch
            </a>
          </div>
        </div>
      )}
    </>
  );
}