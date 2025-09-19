'use client';

import { useEffect, useRef, useState } from 'react';
import { Menu, X } from 'lucide-react';
import { gsap } from 'gsap';
import { cn } from '@/lib/utils';
import CustomLogo from '../../logos/custom-logo';
import { Button } from '../../ui/button';

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
  const navRef = useRef<HTMLElement>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (navRef.current) {
      gsap.fromTo(
        navRef.current,
        { y: -100, opacity: 0 },
        { y: 0, opacity: 1, duration: 1, ease: 'power2.out', delay: 0.5 }
      );
    }
  }, []);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <>
      <nav
        ref={navRef}
        className={cn(
          'fixed top-4 left-1/2 transform -translate-x-1/2 z-50 transition-all duration-300',
          isScrolled
            ? 'bg-black/80 backdrop-blur-md border border-primary/30 shadow-lg shadow-primary/20'
            : 'bg-black/20 backdrop-blur-sm border border-white/10',
          'rounded-full px-6 py-3 max-w-4xl w-full mx-4',
          className
        )}
      >
        <div className="flex items-center justify-between">
          {/* Logo */}
          <a href={homeUrl} className="flex items-center gap-3 group">
            <CustomLogo size="sm" />
            <span className="text-xl font-bold text-white group-hover:text-primary transition-colors duration-300">
              {name}
            </span>
          </a>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            {links.map((link, index) => (
              <a
                key={index}
                href={link.href}
                className="text-white/80 hover:text-primary transition-all duration-300 font-medium relative group"
              >
                {link.text}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-primary to-primary/60 group-hover:w-full transition-all duration-300" />
              </a>
            ))}
          </div>

          {/* CTA Button */}
          <div className="hidden md:block">
            <Button
              className="btn-gradient-primary text-white px-6 py-2 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              asChild
            >
              <a href="/contact">Get In Touch</a>
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={toggleMobileMenu}
            className="md:hidden text-white hover:text-primary transition-colors duration-300"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="relative z-50 flex flex-col items-center justify-center h-full gap-8">
            {links.map((link, index) => (
              <a
                key={index}
                href={link.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className="text-2xl text-white hover:text-primary transition-colors duration-300 font-medium"
              >
                {link.text}
              </a>
            ))}
            <Button
              className="btn-gradient-primary text-white px-8 py-3 rounded-full font-semibold text-lg"
              asChild
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <a href="/contact">Get In Touch</a>
            </Button>
          </div>
        </div>
      )}
    </>
  );
}