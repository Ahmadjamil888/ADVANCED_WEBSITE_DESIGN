'use client';

import Link from 'next/link';
import { Calendar } from 'lucide-react';

export default function BookMeetingButton() {
  return (
    <Link
      href="/contact"
      className="fixed bottom-6 left-6 z-50 bg-black border-2 border-white text-white px-6 py-3 rounded-full shadow-2xl hover:bg-gray-900 hover:border-gray-300 transition-all duration-300 transform hover:scale-105 flex items-center gap-2 font-semibold"
    >
      <Calendar className="w-5 h-5" />
      Book a Meeting
    </Link>
  );
}