"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  Plus, Menu, User, Sparkles, ArrowUp, Trash2, Edit3, 
  MoreHorizontal, Image, Mic, Paperclip, History, 
  MessageSquare, Settings, LogOut, Download
} from 'lucide-react';
import Link from 'next/link';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  attachments?: Attachment[];
}

interface Attachment {
  id: string;
  name: string;
  type: 'image' | 'audio' | 'file';
  url: string;
  size: number;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// AI system prompt to define Zehan AI's personality and knowledge
const SYSTEM_PROMPT = `You are Zehan GPT, Pakistan's most advanced AI assistant created by Zehan X Technologies. You are knowledgeable about:

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

interface ChatHistoryItemProps {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRename: (newTitle: string) => void;
}

const ChatHistoryItem: React.FC<ChatHistoryItemProps> = ({
  session,
  isActive,
  onSelect,
  onDelete,
  onRename
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [title, setTitle] = useState(session.title);

  const handleRename = () => {
    onRename(title);
    setIsEditing(false);
  };

  return (
    <div className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer ${
      isActive ? 'bg-gray-800' : 'hover:bg-gray-800'
    }`}>
      <button onClick={onSelect} className="flex-1 text-left">
        <span className="text-sm text-gray-300">{session.title}</span>
      </button>
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
        <button
          onClick={() => setIsEditing(true)}
          className="p-1 text-gray-400 hover:text-white"
        >
          <Edit3 className="w-3 h-3" />
        </button>
        <button
          onClick={onDelete}
          className="p-1 text-gray-400 hover:text-white"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
};

export default function ZehanGPT() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Load chat sessions from localStorage on component mount
  useEffect(() => {
    const savedSessions = localStorage.getItem('zehan-chat-sessions');
    if (savedSessions) {
      const sessions = JSON.parse(savedSessions).map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        updatedAt: new Date(session.updatedAt),
        messages: session.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
      }));
      setChatSessions(sessions);
    }
  }, []);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('zehan-chat-sessions', JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
      return "Zehan X Technologies is Pakistan's leading AI and web development company. We specialize in creating intelligent solutions using cutting-edge technologies like Next.js, React, and advanced machine learning models. Our team transforms businesses through custom AI applications and modern web development.";
    }
    
    if (input.includes('ai') || input.includes('artificial intelligence') || input.includes('machine learning')) {
      return "We offer comprehensive AI services including custom machine learning models, predictive analytics, natural language processing, computer vision, and intelligent automation systems. Our AI solutions are designed to solve real business problems and drive measurable results.";
    }
    
    if (input.includes('web') || input.includes('website') || input.includes('development') || input.includes('next.js') || input.includes('react')) {
      return "Our web development expertise includes modern frameworks like Next.js, React, and TypeScript. We build scalable, performant web applications with excellent user experiences. From simple business websites to complex enterprise applications, we deliver solutions that drive growth.";
    }
    
    // General AI assistant responses
    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
      return "Hello! I'm Zehan GPT, your intelligent assistant from Zehan X Technologies. I'm here to help you with AI, technology, business questions, and much more. What would you like to know?";
    }
    
    // Default intelligent response
    return `That's an interesting question about "${userInput}". As Zehan GPT, I'm designed to help with AI development, machine learning, web development, and business topics. Could you be more specific about what aspect you'd like to know more about? I can provide detailed information about our services, technologies, or how we can help solve your challenges.`;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const aiResponse = await generateAIResponse(currentInput);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        isUser: false,
        timestamp: new Date()
      };

      const finalMessages = [...newMessages, aiMessage];
      setMessages(finalMessages);

      // Save or update chat session
      saveCurrentSession(finalMessages);
    } catch (err) {
      console.error('Error generating response:', err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I apologize, but I encountered an error. Please try again or contact our support team.',
        isUser: false,
        timestamp: new Date()
      };
      const finalMessages = [...newMessages, errorMessage];
      setMessages(finalMessages);
      saveCurrentSession(finalMessages);
    } finally {
      setIsLoading(false);
    }
  };

  const saveCurrentSession = (sessionMessages: Message[]) => {
    if (sessionMessages.length === 0) return;

    const now = new Date();
    const sessionTitle = generateSessionTitle(sessionMessages[0].text);

    if (currentSessionId) {
      // Update existing session
      setChatSessions(prev => prev.map(session => 
        session.id === currentSessionId 
          ? { ...session, messages: sessionMessages, updatedAt: now, title: sessionTitle }
          : session
      ));
    } else {
      // Create new session
      const newSessionId = Date.now().toString();
      const newSession: ChatSession = {
        id: newSessionId,
        title: sessionTitle,
        messages: sessionMessages,
        createdAt: now,
        updatedAt: now
      };
      setChatSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(newSessionId);
    }
  };

  const generateSessionTitle = (firstMessage: string): string => {
    const words = firstMessage.split(' ').slice(0, 4).join(' ');
    return words.length > 30 ? words.substring(0, 30) + '...' : words;
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputMessage('');
    setIsLoading(false);
    setCurrentSessionId(null);
    setSidebarOpen(false);
  };

  const loadChatSession = (sessionId: string) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setMessages(session.messages);
      setCurrentSessionId(sessionId);
      setSidebarOpen(false);
    }
  };

  const deleteChatSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(s => s.id !== sessionId));
    if (currentSessionId === sessionId) {
      handleNewChat();
    }
  };

  const renameChatSession = (sessionId: string, newTitle: string) => {
    setChatSessions(prev => prev.map(session => 
      session.id === sessionId 
        ? { ...session, title: newTitle }
        : session
    ));
  };

  const exportChatHistory = () => {
    const dataStr = JSON.stringify(chatSessions, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `zehan-gpt-history-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      // Handle file upload logic here
      console.log('File uploaded:', file.name);
      // You can add file processing logic here
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        // Handle image upload logic here
        console.log('Image uploaded:', file.name);
        // You can add image processing logic here
      }
    }
  };

  const startVoiceRecording = () => {
    setIsRecording(true);
    // Add voice recording logic here
    console.log('Voice recording started');
  };

  const stopVoiceRecording = () => {
    setIsRecording(false);
    // Add voice recording stop logic here
    console.log('Voice recording stopped');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  };

  const suggestedPrompts = [
    "Explain machine learning in simple terms",
    "How can AI transform my business?",
    "What services does Zehan X offer?",
    "Latest trends in web development"
  ];

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900 flex overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-gray-900 transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className="text-white font-semibold">Zehan GPT</span>
            </Link>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden text-gray-400 hover:text-white"
            >
              ×
            </button>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={handleNewChat}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              New chat
            </button>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto px-4">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Chat History</div>
              <button
                onClick={exportChatHistory}
                className="text-gray-500 hover:text-gray-300 p-1"
                title="Export chat history"
              >
                <Download className="w-3 h-3" />
              </button>
            </div>
            
            {chatSessions.length === 0 ? (
              <div className="text-xs text-gray-600 text-center py-4">
                No chat history yet
              </div>
            ) : (
              <div className="space-y-1">
                {chatSessions.map((session) => (
                  <ChatHistoryItem
                    key={session.id}
                    session={session}
                    isActive={currentSessionId === session.id}
                    onSelect={() => loadChatSession(session.id)}
                    onDelete={() => deleteChatSession(session.id)}
                    onRename={(newTitle) => renameChatSession(session.id, newTitle)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* User Section */}
          <div className="p-4 border-t border-gray-700">
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
              >
                <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <span className="flex-1 text-left">User</span>
                <MoreHorizontal className="w-4 h-4" />
              </button>

              {/* User Menu */}
              {showUserMenu && (
                <div className="absolute bottom-full left-0 right-0 mb-2 bg-gray-800 border border-gray-700 rounded-lg shadow-lg">
                  <button 
                    onClick={() => {
                      setChatSessions([]);
                      handleNewChat();
                      setShowUserMenu(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-t-lg transition-colors"
                  >
                    <History className="w-4 h-4" />
                    Clear All History
                  </button>
                  <button 
                    onClick={exportChatHistory}
                    className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Export History
                  </button>
                  <Link 
                    href="/"
                    className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-b-lg transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                    Back to Home
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between lg:justify-end">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            Zehan GPT
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="max-w-2xl mx-auto text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
                  What can I help with?
                </h1>
                <p className="text-gray-600 dark:text-gray-300 mb-8">
                  I'm Zehan GPT, Pakistan's most advanced AI assistant. Ask me anything about AI, technology, business, or get help with your projects.
                </p>
                
                {/* Suggested Prompts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                  {suggestedPrompts.map((prompt, index) => (
                    <button
                      key={index}
                      onClick={() => setInputMessage(prompt)}
                      className="p-4 text-left bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      <div className="text-sm text-gray-900 dark:text-white font-medium">
                        {prompt}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* Chat Messages */
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-3xl mx-auto px-4 py-6">
                {messages.map((message) => (
                  <div key={message.id} className="mb-8">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0">
                        {message.isUser ? (
                          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                            <User className="w-4 h-4 text-white" />
                          </div>
                        ) : (
                          <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                            <Sparkles className="w-4 h-4 text-white" />
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                          {message.isUser ? 'You' : 'Zehan GPT'}
                        </div>
                        <div className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                          {message.text}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Loading Indicator */}
                {isLoading && (
                  <div className="mb-8">
                    <div className="flex items-start gap-4">
                      <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                        <Sparkles className="w-4 h-4 text-white" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                          Zehan GPT
                        </div>
                        <div className="flex items-center gap-2 text-gray-500">
                          <div className="flex gap-1">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                          </div>
                          <span className="text-sm">Thinking...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <div className="max-w-3xl mx-auto p-4">
              <div className="relative">
                {/* File Upload Inputs */}
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileUpload}
                  className="hidden"
                  accept=".pdf,.doc,.docx,.txt,.csv,.xlsx"
                />
                <input
                  ref={audioInputRef}
                  type="file"
                  onChange={handleImageUpload}
                  className="hidden"
                  accept="image/*"
                />

                <div className="flex items-end gap-2">
                  {/* Attachment Buttons */}
                  <div className="flex items-center gap-1 mb-2">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                      title="Upload file"
                    >
                      <Paperclip className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => audioInputRef.current?.click()}
                      className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                      title="Upload image"
                    >
                      <Image className="w-4 h-4" />
                    </button>
                    <button
                      onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
                      className={`p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors ${
                        isRecording 
                          ? 'text-red-500 bg-red-50 dark:bg-red-900/20' 
                          : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                      }`}
                      title={isRecording ? "Stop recording" : "Start voice recording"}
                    >
                      <Mic className={`w-4 h-4 ${isRecording ? 'animate-pulse' : ''}`} />
                    </button>
                  </div>

                  {/* Text Input */}
                  <div className="flex-1 relative">
                    <textarea
                      ref={textareaRef}
                      value={inputMessage}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyDown}
                      placeholder="Message Zehan GPT..."
                      disabled={isLoading}
                      rows={1}
                      className="w-full resize-none rounded-2xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-3 pr-12 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{ minHeight: '48px', maxHeight: '200px' }}
                    />
                    <button
                      onClick={handleSendMessage}
                      disabled={!inputMessage.trim() || isLoading}
                      className="absolute right-2 bottom-2 p-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {isLoading ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <ArrowUp className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
                Zehan GPT can make mistakes. Consider checking important information.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}