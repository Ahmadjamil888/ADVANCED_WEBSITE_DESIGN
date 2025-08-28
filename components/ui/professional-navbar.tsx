'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Menu, X, Bell } from 'lucide-react';
import { Button } from './button';

export default function ProfessionalNavbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav className="relative bg-black/90 backdrop-blur-md border-b border-gray-800 after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-px after:bg-white/20">
      <div className="mx-auto max-w-7xl px-2 sm:px-6 lg:px-8">
        <div className="relative flex h-16 items-center justify-between">
          {/* Mobile menu button */}
          <div className="absolute inset-y-0 left-0 flex items-center sm:hidden">
            <button
              type="button"
              onClick={toggleMobileMenu}
              className="relative inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-white/5 hover:text-white focus:outline-2 focus:-outline-offset-1 focus:outline-blue-500"
            >
              <span className="absolute -inset-0.5"></span>
              <span className="sr-only">Open main menu</span>
              {isMobileMenuOpen ? (
                <X className="size-6" />
              ) : (
                <Menu className="size-6" />
              )}
            </button>
          </div>

          {/* Logo and navigation */}
          <div className="flex flex-1 items-center justify-center sm:items-stretch sm:justify-start">
            <div className="flex shrink-0 items-center">
              <Link href="/" className="text-xl font-bold text-white">
                ZEHANX
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:block">
              <div className="flex space-x-4">
                <Link
                  href="/"
                  className="rounded-md bg-gray-900/80 px-3 py-2 text-sm font-medium text-white"
                  aria-current="page"
                >
                  Home
                </Link>
                <Link
                  href="/services"
                  className="rounded-md px-3 py-2 text-sm font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  Services
                </Link>
                <Link
                  href="/about"
                  className="rounded-md px-3 py-2 text-sm font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  About
                </Link>
                <Link
                  href="/portfolio"
                  className="rounded-md px-3 py-2 text-sm font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  Portfolio
                </Link>
                <Link
                  href="/blog"
                  className="rounded-md px-3 py-2 text-sm font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  Blog
                </Link>
              </div>
            </div>
          </div>

          {/* Right side buttons */}
          <div className="absolute inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0 gap-4">
            <Button
              variant="outline"
              size="sm"
              className="hidden sm:inline-flex border-gray-600 text-gray-300 hover:bg-white/5 hover:text-white"
              asChild
            >
              <Link href="/contact">Get In Touch</Link>
            </Button>
            <Button
              size="sm"
              className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white border-0"
              asChild
            >
              <Link href="/zehan">Get Started</Link>
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMobileMenuOpen && (
        <div className="sm:hidden">
          <div className="space-y-1 px-2 pt-2 pb-3">
            <Link
              href="/"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block rounded-md bg-gray-900/80 px-3 py-2 text-base font-medium text-white"
              aria-current="page"
            >
              Home
            </Link>
            <Link
              href="/services"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block rounded-md px-3 py-2 text-base font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
            >
              Services
            </Link>
            <Link
              href="/about"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block rounded-md px-3 py-2 text-base font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
            >
              About
            </Link>
            <Link
              href="/portfolio"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block rounded-md px-3 py-2 text-base font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
            >
              Portfolio
            </Link>
            <Link
              href="/blog"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block rounded-md px-3 py-2 text-base font-medium text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
            >
              Blog
            </Link>
            <div className="flex flex-col gap-2 mt-4 px-3">
              <Button
                variant="outline"
                size="sm"
                className="border-gray-600 text-gray-300 hover:bg-white/5 hover:text-white"
                asChild
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <Link href="/contact">Get In Touch</Link>
              </Button>
              <Button
                size="sm"
                className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white border-0"
                asChild
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <Link href="/zehan">Get Started</Link>
              </Button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}