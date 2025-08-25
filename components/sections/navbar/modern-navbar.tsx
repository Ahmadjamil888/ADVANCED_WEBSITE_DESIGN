'use client';

import { useEffect, useRef, useState } from 'react';
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
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <>
      <nav
        className={cn(
          'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
          isScrolled ? 'bg-black/90 backdrop-blur-sm' : 'bg-transparent',
          className
        )}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <a href={homeUrl} className="flex items-center">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/25">
                <span className="text-white font-bold text-lg">Z</span>
              </div>
              <span className="ml-3 text-xl font-bold text-white">
                {name}
              </span>
            </a>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              {links.map((link, index) => (
                <a
                  key={index}
                  href={link.href}
                  className="text-white hover:text-blue-300 transition-colors duration-300 font-medium"
                >
                  {link.text}
                </a>
              ))}
            </div>

            {/* CTA Button */}
            <div className="hidden md:block">
              <a
                href="/contact"
                className="px-6 py-2.5 bg-blue-600 text-white font-semibold rounded-lg shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-shadow duration-300"
              >
                Get Started
              </a>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={toggleMobileMenu}
              className="md:hidden text-white"
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/95" />
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
              className="mt-8 px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg text-lg shadow-lg shadow-blue-500/25"
            >
              Get Started
            </a>
          </div>
        </div>
      )}
    </>
  );
}