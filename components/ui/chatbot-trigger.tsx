'use client';

import { useState, useEffect } from 'react';
import { MessageCircle, X } from 'lucide-react';

export default function ChatbotTrigger() {
  const [isVisible, setIsVisible] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Check if Botpress is loaded
    const checkBotpress = () => {
      if (typeof window !== 'undefined') {
        // Check for Botpress webchat
        const botpressElement = document.querySelector('#bp-web-widget');
        const botpressScript = document.querySelector('script[src*="botpress"]');
        
        if (botpressElement || botpressScript) {
          setIsLoaded(true);
          return true;
        }
      }
      return false;
    };

    // Check immediately
    checkBotpress();

    // Check periodically until loaded
    const interval = setInterval(() => {
      if (checkBotpress()) {
        clearInterval(interval);
      }
    }, 1000);

    // Clean up after 30 seconds
    const timeout = setTimeout(() => {
      clearInterval(interval);
    }, 30000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, []);

  const openChat = () => {
    // Try multiple methods to open the chat
    if (typeof window !== 'undefined') {
      // Method 1: Try Botpress webchat API
      if ((window as any).botpressWebChat) {
        (window as any).botpressWebChat.sendEvent({ type: 'show' });
        return;
      }

      // Method 2: Try clicking the Botpress widget
      const botpressWidget = document.querySelector('#bp-web-widget button') as HTMLElement;
      if (botpressWidget) {
        botpressWidget.click();
        return;
      }

      // Method 3: Try finding any chat widget
      const chatWidget = document.querySelector('[data-testid="widget-button"]') as HTMLElement;
      if (chatWidget) {
        chatWidget.click();
        return;
      }

      // Method 4: Fallback - redirect to contact page
      window.location.href = '/contact';
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
      {/* Chat Button */}
      <button
        onClick={openChat}
        className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-110 relative group"
        aria-label="Open AI Chat"
      >
        <MessageCircle className="w-6 h-6" />
        <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center animate-pulse">
          AI
        </span>
        
        {/* Tooltip */}
        <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 bg-gray-900 text-white text-sm px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
          Chat with AI Assistant
          <div className="absolute left-full top-1/2 -translate-y-1/2 border-4 border-transparent border-l-gray-900"></div>
        </div>
      </button>

      {/* Close button */}
      <button
        onClick={() => setIsVisible(false)}
        className="bg-gray-600 hover:bg-gray-700 text-white p-2 rounded-full shadow-lg transition-all duration-300 text-xs"
        aria-label="Hide chat button"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}