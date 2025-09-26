"use client";

import { useState, useEffect } from 'react';
import { Send, Bot, RotateCcw } from 'lucide-react';
import Navbar from '../../components/sections/navbar/default';
import Footer from '../../components/sections/footer/default';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

// AI system prompt to define Zehan AI's personality and knowledge
const SYSTEM_PROMPT = `You are Zehan AI, Pakistan's first advanced AI assistant created by Zehan X Technologies. You are knowledgeable about:

- Artificial Intelligence and Machine Learning
- Web Development (especially Next.js, React, TypeScript)
- Deep Learning and Neural Networks
- Generative AI and Large Language Models
- Business AI Solutions and Automation
- Data Analytics and Predictive Modeling
- Enterprise Software Development
- Digital Marketing and Creative Services
- Video Editing and Content Creation

You should be helpful, professional, and showcase the capabilities of Zehan X Technologies. Keep responses concise but informative. Always maintain a friendly and expert tone. You represent Pakistan's pioneering AI technology.`;

// Removed unused modelPipeline variable

export default function ZehanAI() {
  const initialMessage: Message = {
    id: '1',
    text: 'Hello! I\'m Zehan AI, Pakistan\'s first advanced AI assistant! I\'m powered by cutting-edge technology and can help with complex questions about AI, web development, business solutions, and much more. Ask me anything to experience the future of AI!',
    isUser: false,
    timestamp: new Date()
  };

  const [messages, setMessages] = useState<Message[]>([initialMessage]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');

  // Initialize AI system (using API instead of heavy client-side model)
  useEffect(() => {
    // Set model as ready immediately since we're using API
    setModelStatus('ready');
  }, []);

  const generateAIResponse = async (userInput: string): Promise<string> => {
    try {
      // Use the API endpoint for AI responses
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: userInput }
          ]
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();
      return data.message || generateFallbackResponse(userInput);
    } catch (err) {
      console.error('AI Response Error:', err);
      return generateFallbackResponse(userInput);
    }
  };

  const generateFallbackResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();
    
    // Zehan X Technologies specific responses
    if (input.includes('zehan') || input.includes('company') || input.includes('about')) {
      return "Zehan X Technologies is a leading AI and web development company. We specialize in creating intelligent solutions using cutting-edge technologies like Next.js, React, and advanced machine learning models. Our team transforms businesses through custom AI applications and modern web development.";
    }
    
    if (input.includes('ai') || input.includes('artificial intelligence') || input.includes('machine learning')) {
      return "We offer comprehensive AI services including custom machine learning models, predictive analytics, natural language processing, computer vision, and intelligent automation systems. Our AI solutions are designed to solve real business problems and drive measurable results.";
    }
    
    if (input.includes('web') || input.includes('website') || input.includes('development') || input.includes('next.js') || input.includes('react')) {
      return "Our web development expertise includes modern frameworks like Next.js, React, and TypeScript. We build scalable, performant web applications with excellent user experiences. From simple business websites to complex enterprise applications, we deliver solutions that drive growth.";
    }
    
    if (input.includes('service') || input.includes('help') || input.includes('offer')) {
      return "We offer AI & Machine Learning solutions, Next.js development, deep learning systems, full-stack web applications, intelligent chatbots, performance optimization, data analytics, and enterprise security. Each solution is customized to meet your specific business needs.";
    }
    
    if (input.includes('price') || input.includes('cost') || input.includes('pricing')) {
      return "Our pricing is customized based on your specific requirements and project scope. We offer competitive rates for both AI development and web development services. Contact us for a detailed quote tailored to your needs.";
    }
    
    if (input.includes('contact') || input.includes('reach') || input.includes('get in touch')) {
      return "You can reach out to us through our contact page or email us directly. We're always happy to discuss your project requirements and how our AI and web development expertise can help transform your business.";
    }
    
    // General AI assistant responses
    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
      return "Hello! I'm Zehan AI, your intelligent assistant from Zehan X Technologies. I'm here to help you learn about our AI and web development services. What would you like to know?";
    }
    
    if (input.includes('thank') || input.includes('thanks')) {
      return "You're welcome! I'm glad I could help. If you have any more questions about our AI solutions or web development services, feel free to ask!";
    }
    
    // Default intelligent response
    return `That's an interesting question about "${userInput}". As Zehan AI, I'm designed to help with AI development, machine learning, and web development topics. Could you be more specific about what aspect you'd like to know more about? I can provide detailed information about our services, technologies, or how we can help solve your business challenges.`;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    try {
      const aiResponse = await generateAIResponse(currentInput);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error generating response:', err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I apologize, but I encountered an error. Please try again or contact our support team.',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
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
                <p className="text-sm text-primary font-medium">Pakistan's First Advanced AI Assistant</p>
              </div>
              <p className="text-lg text-muted-foreground">How can I help you today? I can answer complex questions about AI, technology, business, and more!</p>
            </div>

            {/* Model Status */}
            {modelStatus === 'loading' && (
              <div className="text-center mb-4">
                <div className="inline-flex items-center gap-2 text-muted-foreground text-sm">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                  Loading AI model...
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="w-full max-w-xl mx-auto">
              <div className="relative">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={modelStatus === 'ready' ? "Message Zehan AI" : "Loading AI model..."}
                  disabled={isLoading || modelStatus !== 'ready'}
                  className="w-full px-4 py-4 pr-20 bg-card border border-border rounded-2xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed text-base"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isLoading || modelStatus !== 'ready'}
                    className="p-2 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="size-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            {modelStatus === 'ready' && (
              <div className="flex flex-wrap justify-center gap-2 text-sm">
                <button
                  onClick={() => setInputMessage("Tell me about Zehan X Technologies and why it's Pakistan's leading AI company")}
                  className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
                >
                  About Zehanx
                </button>
                <button
                  onClick={() => setInputMessage("Explain the difference between machine learning and deep learning with examples")}
                  className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
                >
                  AI Concepts
                </button>
                <button
                  onClick={() => setInputMessage("How can AI transform my business and what's the ROI?")}
                  className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
                >
                  Business AI
                </button>
                <button
                  onClick={() => setInputMessage("What are the latest trends in generative AI and how can I implement them?")}
                  className="px-4 py-2 bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-lg transition-colors"
                >
                  Gen AI Trends
                </button>
              </div>
            )}

            {modelStatus === 'error' && (
              <div className="text-center text-red-500 text-sm">
                Failed to load AI model. Using fallback responses.
              </div>
            )}
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
                  onKeyDown={handleKeyDown}
                  placeholder={modelStatus === 'ready' ? "Message Zehan AI" : "Loading AI model..."}
                  disabled={isLoading || modelStatus !== 'ready'}
                  className="w-full px-4 py-4 pr-20 bg-card border border-border rounded-2xl text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed text-base"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isLoading || modelStatus !== 'ready'}
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