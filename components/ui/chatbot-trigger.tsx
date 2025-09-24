'use client';

import { useState, useEffect } from 'react';
import { MessageCircle } from 'lucide-react';

// Define types for Botpress
interface BotpressWebChat {
  sendEvent: (event: { type: string }) => void;
}

declare global {
  interface Window {
    botpressWebChat?: BotpressWebChat;
  }
}

export default function ChatbotTrigger() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Check if Botpress is loaded
    const checkBotpress = () => {
      if (typeof window !== 'undefined' && window.botpressWebChat) {
        setIsLoaded(true);
      }
    };

    // Check immediately
    checkBotpress();

    // Check periodically until loaded
    const interval = setInterval(() => {
      checkBotpress();
      if (isLoaded) {
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isLoaded]);

  const openChat = () => {
    if (typeof window !== 'undefined' && window.botpressWebChat) {
      window.botpressWebChat.sendEvent({ type: 'show' });
    } else {
      console.log('Botpress not loaded yet');
    }
  };

  return (
    <button
      onClick={openChat}
      className="fixed bottom-6 right-6 z-50 bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-110"
      aria-label="Open AI Chat"
    >
      <MessageCircle className="w-6 h-6" />
      <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center animate-pulse">
        AI
      </span>
    </button>
  );
}