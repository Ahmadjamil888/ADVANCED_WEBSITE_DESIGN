'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { Menu, X, Home, Zap, BookOpen, Settings, Download, LogOut } from 'lucide-react';
import { supabase } from '@/lib/supabase';

export default function ZehanxAILayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const pathname = usePathname();
  const router = useRouter();

  const handleSignOut = async () => {
    try {
      if (supabase) {
        await supabase.auth.signOut();
      }
      router.push('/login');
    } catch (error) {
      console.error('Sign out error:', error);
      router.push('/login');
    }
  };

  const navItems = [
    { href: '/zehanx-ai', label: 'Dashboard', icon: Home },
    { href: '/zehanx-ai/generator', label: 'Model Generator', icon: Zap },
    { href: '/zehanx-ai/datasets', label: 'Datasets', icon: BookOpen },
    { href: '/zehanx-ai/models', label: 'My Models', icon: Download },
    { href: '/zehanx-ai/settings', label: 'Settings', icon: Settings },
  ];

  const isActive = (href: string) => {
    if (href === '/zehanx-ai') {
      return pathname === '/zehanx-ai';
    }
    return pathname.startsWith(href);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-800/50 backdrop-blur border-b border-slate-700 sticky top-0 z-40">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div>
              <h1 className="text-2xl font-bold text-white">Zehanx AI</h1>
              <p className="text-sm text-slate-400">AI Model Generation Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
            >
              ← Back to Home
            </Link>
            <button
              onClick={handleSignOut}
              className="px-4 py-2 text-slate-300 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors flex items-center gap-2"
              title="Sign out"
            >
              <LogOut size={18} />
              <span className="text-sm">Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? 'w-64' : 'w-0'
          } bg-slate-800/30 border-r border-slate-700 transition-all duration-300 overflow-hidden`}
        >
          <nav className="p-4 space-y-2">
            {navItems.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  isActive(href)
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/50'
                    : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                }`}
              >
                <Icon size={20} />
                <span className="font-medium">{label}</span>
              </Link>
            ))}
          </nav>

          {/* Sidebar Footer */}
          <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-700 bg-slate-800/50">
            <div className="text-xs text-slate-400 space-y-1">
              <p className="font-semibold text-slate-300">Quick Stats</p>
              <p>Models: 0</p>
              <p>Datasets: 0</p>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
