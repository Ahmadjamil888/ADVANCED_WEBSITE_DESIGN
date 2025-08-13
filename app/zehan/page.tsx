"use client";

import { useState } from 'react';
import { Send, Bot, User, RotateCcw } from 'lucide-react';
import Navbar from '../../components/sections/navbar/default';
import Footer from '../../components/sections/footer/default';

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
  const initialMessage: Message = {
    id: '1',
    text: 'Hello! I\'m Zehan AI, a demonstration of our advanced AI capabilities. Ask me anything to see how our technology works!',
    isUser: false,
    timestamp: new Date()
  };

  const [messages, setMessages] = useState<Message[]>([initialMessage]);
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

  const handleResetChat = () => {
    setMessages([{
      ...initialMessage,
      id: Date.now().toString(),
      timestamp: new Date()
    }]);
    setInputMessage('');
    setIsLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Navbar */}
      <Navbar />

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 py-8">
        {messages.length === 1 && !isLoading ? (
          /* Initial Welcome Screen */
          <div className="w-full max-w-2xl mx-auto text-center space-y-8">
            {/* Logo and Welcome */}
            <div className="space-y-4">
              <div className="flex items-center justify-center gap-3 mb-6">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                  <Bot className="size-6 text-primary" />
                </div>
                <h1 className="text-2xl font-bold text-foreground">Hi, I'm Zehan AI.</h1>
              </div>
              <p className="text-lg text-muted-foreground">How can I help you today?</p>
            </div>

            {/* Input Area */}
            <div className="w-full max-w-xl mx-auto">
              <div className="relative">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Message Zehan AI"
                  disabled={isLoading}
                  className="w-full px-4 py-4 pr-20 bg-card border border-border rounded-2xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed text-base"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isLoading}
                    className="p-2 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="size-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap justify-center gap-2 text-sm">
              <button
                onClick={() => setInputMessage("Tell me about Zehan X Technologies")}
                className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
              >
                About Zehanx
              </button>
              <button
                onClick={() => setInputMessage("What AI services do you offer?")}
                className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
              >
                AI Services
              </button>
              <button
                onClick={() => setInputMessage("How can AI help my business?")}
                className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
              >
                Business AI
              </button>
            </div>
          </div>
        ) : (
          /* Chat Conversation View */
          <div className="w-full max-w-4xl mx-auto flex flex-col h-full">
            {/* Chat Header */}
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Bot className="size-4 text-primary" />
                </div>
                <h1 className="text-lg font-semibold text-foreground">Zehan AI</h1>
              </div>
              <button
                onClick={handleResetChat}
                disabled={isLoading}
                className="flex items-center gap-2 px-3 py-2 text-sm bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RotateCcw className="size-4" />
                New Chat
              </button>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto space-y-4 mb-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                      message.isUser
                        ? 'bg-primary text-primary-foreground ml-auto'
                        : 'bg-card border border-border text-foreground mr-auto'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {!message.isUser && (
                        <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <Bot className="size-3 text-primary" />
                        </div>
                      )}
                      <div className="flex-1">
                        <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.text}</p>
                        <span className="text-xs opacity-60 mt-2 block">
                          {message.timestamp.toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-card border border-border max-w-xs lg:max-w-md px-4 py-3 rounded-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                        <Bot className="size-3 text-primary" />
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">Zehan AI is thinking</span>
                        <div className="flex gap-1">
                          <div className="w-1 h-1 bg-primary rounded-full animate-bounce"></div>
                          <div className="w-1 h-1 bg-primary rounded-full animate-bounce delay-100"></div>
                          <div className="w-1 h-1 bg-primary rounded-full animate-bounce delay-200"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="w-full">
              <div className="relative">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Message Zehan AI"
                  disabled={isLoading}
                  className="w-full px-4 py-4 pr-20 bg-card border border-border rounded-2xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed text-base"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isLoading}
                    className="p-2 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="size-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
}