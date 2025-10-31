"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

interface Chat {
  id: string;
  title: string;
  mode: string;
  updated_at: string;
  is_pinned: boolean;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  tokens_used?: number;
}

export default function AIWorkspace() {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();
  
  // State management
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMode, setCurrentMode] = useState<string>('chat');
  const [isLoading, setIsLoading] = useState(false);
  const [contextPanelOpen, setContextPanelOpen] = useState(true);

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
      return;
    }

    if (user) {
      loadChats();
    }
  }, [user, loading, router]);

  const loadChats = async () => {
    if (!supabase || !user) return;

    const { data, error } = await supabase
      .from('chats')
      .select('*')
      .eq('user_id', user.id)
      .order('updated_at', { ascending: false });

    if (data && !error) {
      setChats(data);
      if (data.length > 0 && !currentChat) {
        setCurrentChat(data[0]);
        loadMessages(data[0].id);
      }
    }
  };

  const loadMessages = async (chatId: string) => {
    if (!supabase) return;

    const { data, error } = await supabase
      .from('messages')
      .select('*')
      .eq('chat_id', chatId)
      .order('created_at', { ascending: true });

    if (data && !error) {
      setMessages(data);
    }
  };

  const createNewChat = async () => {
    if (!supabase || !user) return;

    const { data, error } = await supabase
      .from('chats')
      .insert({
        user_id: user.id,
        title: 'Untitled Chat',
        mode: currentMode
      })
      .select()
      .single();

    if (data && !error) {
      setChats(prev => [data, ...prev]);
      setCurrentChat(data);
      setMessages([]);
    }
  };

  const sendMessage = async (content: string) => {
    if (!currentChat || !supabase || !user || isLoading) return;

    setIsLoading(true);

    // Add user message to UI immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      // Save user message to database
      await supabase.from('messages').insert({
        chat_id: currentChat.id,
        role: 'user',
        content
      });

      // Generate AI response
      const response = await fetch('/api/ai-workspace/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: currentChat.id,
          prompt: content,
          mode: currentChat.mode,
          userId: user.id
        })
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Add AI response to UI
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        created_at: new Date().toISOString(),
        tokens_used: data.tokens_used
      };
      setMessages(prev => [...prev, aiMessage]);

      // Save AI message to database
      await supabase.from('messages').insert({
        chat_id: currentChat.id,
        role: 'assistant',
        content: data.response,
        tokens_used: data.tokens_used,
        model_used: data.model_used
      });

      // Update chat timestamp
      await supabase
        .from('chats')
        .update({ updated_at: new Date().toISOString() })
        .eq('id', currentChat.id);

    } catch (error: any) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: `Error: ${error.message}`,
        created_at: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-white">
        <div className="text-lg text-gray-600">Loading AI Workspace...</div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="flex h-screen w-screen bg-white overflow-hidden" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}>
      {/* Sidebar - ChatGPT Style */}
      <div className="w-64 bg-gray-900 text-white flex flex-col">
        <div className="p-4">
          <h1 className="text-white text-xl font-semibold">AI Workspace</h1>
          <p className="text-gray-300 text-sm mt-2">Generate, train, and deploy AI models</p>
          
          <button
            onClick={createNewChat}
            className="w-full mt-4 flex items-center justify-center space-x-2 border border-gray-600 text-white px-3 py-2 rounded-md hover:bg-gray-800 transition-colors text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New chat</span>
          </button>
        </div>
        
        <div className="flex-1 p-4">
          <div className="text-gray-400 text-sm">Recent chats will appear here</div>
        </div>
        
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={signOut}
            className="w-full flex items-center space-x-2 text-gray-300 hover:text-white px-3 py-2 rounded-md hover:bg-gray-800 transition-colors text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            <span>Sign out</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col bg-white">
        {/* Header */}
        <div className="border-b border-gray-200 p-4">
          <h2 className="text-lg font-semibold text-gray-900">AI Model Generator</h2>
          <p className="text-gray-500 text-sm">Describe the AI model you want to create</p>
        </div>
        
        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {messages.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <img src="/logo.jpg" alt="AI" className="w-10 h-10 rounded-full" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Ready to create amazing AI models?
                </h3>
                <p className="text-gray-500 mb-6">
                  I can help you generate, train, and deploy custom AI models. Just describe what you want to build!
                </p>
                
                <div className="grid grid-cols-1 gap-3">
                  <button className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors">
                    <div className="font-medium text-gray-900">🎯 Text Classification</div>
                    <div className="text-sm text-gray-500">Create a sentiment analysis model</div>
                  </button>
                  <button className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors">
                    <div className="font-medium text-gray-900">🖼️ Image Classification</div>
                    <div className="text-sm text-gray-500">Detect and classify objects in images</div>
                  </button>
                  <button className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors">
                    <div className="font-medium text-gray-900">🤖 Chatbot Model</div>
                    <div className="text-sm text-gray-500">Build a conversational AI assistant</div>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-6">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-6`}>
                  <div className={`max-w-3xl rounded-2xl px-4 py-3 ${
                    message.role === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-900'
                  }`}>
                    <div className="whitespace-pre-wrap break-words">
                      {message.content}
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start mb-6">
                  <div className="bg-gray-100 rounded-2xl px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm text-gray-500">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Input Area */}
          <div className="border-t border-gray-200 p-4">
            <div className="max-w-4xl mx-auto">
              <div className="relative bg-white border border-gray-300 rounded-2xl shadow-sm focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500">
                <textarea
                  placeholder="Describe the AI model you want to create (e.g., 'Create a sentiment analysis model using BERT...')"
                  className="w-full resize-none border-none outline-none text-gray-900 placeholder-gray-500 p-4"
                  style={{ minHeight: '60px', maxHeight: '200px' }}
                  disabled={isLoading}
                />
                <div className="flex justify-end p-3 pt-0">
                  <button
                    disabled={isLoading}
                    className="p-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl transition-all"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="text-center mt-3">
                <p className="text-xs text-gray-500">
                  zehanx AI can generate, train, and deploy custom AI models. Always verify generated code before training.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}