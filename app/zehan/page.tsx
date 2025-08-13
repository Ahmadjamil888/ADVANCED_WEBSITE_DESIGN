"use client";

import { useEffect, useState } from 'react';
import { pipeline } from '@huggingface/transformers';
import { Send, Bot, User } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

let pipe: any;

export default function ZehanAI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPipelineReady, setIsPipelineReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        pipe = await pipeline('text-generation', 'Xenova/distilgpt2');
        setIsPipelineReady(true);
        // Add welcome message
        setMessages([{
          id: '1',
          text: 'Hello! I\'m Zehan AI, powered by DistilGPT-2. How can I help you today?',
          isUser: false,
          timestamp: new Date()
        }]);
      } catch (error) {
        console.error('Failed to load AI model:', error);
      }
    })();
  }, []);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !isPipelineReady || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const output = await pipe(inputMessage, { 
        max_length: 50,
        temperature: 0.7,
        do_sample: true,
        pad_token_id: 50256
      });
      
      const aiReply = output[0].generated_text;
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiReply,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error generating response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
          {!isPipelineReady && (
            <div className="text-center py-8">
              <div className="inline-flex items-center gap-2 text-muted-foreground">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                Loading AI model...
              </div>
            </div>
          )}

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
              placeholder={isPipelineReady ? "Type your message..." : "Loading AI model..."}
              disabled={!isPipelineReady || isLoading}
              className="flex-1 px-3 py-2 border border-border rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || !isPipelineReady || isLoading}
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