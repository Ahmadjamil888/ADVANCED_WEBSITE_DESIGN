"use client";

import { useState } from 'react';
import { Send, Bot, User } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

// Demo responses for the AI
const demoResponses = [
  "That's an interesting question! As Zehan AI, I'm designed to help with various tasks and provide intelligent responses.",
  "I'm here to demonstrate the capabilities of Zehan X Technologies' AI solutions. What would you like to know?",
  "Great question! Our AI technology can be customized for your specific business needs and requirements.",
  "I'm a demo version of Zehan AI. The full version can be integrated into your applications with advanced capabilities.",
  "Thank you for trying Zehan AI! This demonstrates our expertise in artificial intelligence and machine learning.",
  "As an AI assistant, I can help with various tasks. This is just a preview of what's possible with our technology.",
  "Zehan X Technologies specializes in creating custom AI solutions like this one for businesses worldwide.",
  "This chat interface showcases our ability to create intelligent, responsive AI applications for any industry."
];

export default function ZehanAI() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m Zehan AI, a demonstration of our advanced AI capabilities. Ask me anything to see how our technology works!',
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // Simulate AI processing time
    setTimeout(() => {
      const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: randomResponse,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1000 + Math.random() * 2000); // Random delay between 1-3 seconds
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const currentYear = new Date().getFullYear();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Navbar */}
      <nav className="bg-card border-b border-border px-4 py-3 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold text-primary">Zehan AI</h1>
          <span className="text-sm text-muted-foreground">Developed by Zehanx Team</span>
        </div>
      </nav>

      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.isUser
                    ? 'bg-primary text-primary-foreground ml-auto'
                    : 'bg-muted text-muted-foreground mr-auto'
                }`}
              >
                <div className="flex items-start gap-2">
                  {!message.isUser && (
                    <Bot className="size-4 mt-0.5 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                    <span className="text-xs opacity-70 mt-1 block">
                      {message.timestamp.toLocaleTimeString([], { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </span>
                  </div>
                  {message.isUser && (
                    <User className="size-4 mt-0.5 flex-shrink-0" />
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted text-muted-foreground max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
                <div className="flex items-center gap-2">
                  <Bot className="size-4" />
                  <div className="flex items-center gap-1">
                    <span className="text-sm">Zehan AI is typing</span>
                    <div className="flex gap-1">
                      <div className="w-1 h-1 bg-current rounded-full animate-bounce"></div>
                      <div className="w-1 h-1 bg-current rounded-full animate-bounce delay-100"></div>
                      <div className="w-1 h-1 bg-current rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-border p-4 bg-card">
          <div className="flex gap-2 max-w-4xl mx-auto">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              disabled={isLoading}
              className="flex-1 px-3 py-2 border border-border rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="size-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-card border-t border-border py-4 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-sm text-muted-foreground">
            © {currentYear} Zehanx Technologies. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}