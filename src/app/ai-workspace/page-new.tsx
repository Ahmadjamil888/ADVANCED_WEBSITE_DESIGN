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
  eventId?: string;
}

export default function AIWorkspace() {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();
  
  // State management
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingModels, setPendingModels] = useState<Set<string>>(new Set());
  const [inputValue, setInputValue] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showProfileMenu, setShowProfileMenu] = useState(false);

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

    try {
      const { data, error } = await supabase
        .from('chats')
        .insert({
          user_id: user.id,
          title: 'New Chat',
          mode: 'models'
        })
        .select()
        .single();

      if (data && !error) {
        setChats(prev => [data, ...prev]);
        setCurrentChat(data);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  const deleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!supabase) return;

    try {
      await supabase.from('messages').delete().eq('chat_id', chatId);
      await supabase.from('chats').delete().eq('id', chatId);
      
      setChats(prev => prev.filter(chat => chat.id !== chatId));
      
      if (currentChat?.id === chatId) {
        const remainingChats = chats.filter(chat => chat.id !== chatId);
        if (remainingChats.length > 0) {
          setCurrentChat(remainingChats[0]);
          loadMessages(remainingChats[0].id);
        } else {
          setCurrentChat(null);
          setMessages([]);
        }
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const sendMessage = async (content: string) => {
    if (!supabase || !user || isLoading) return;

    // Create a new chat if none exists
    let chatToUse = currentChat;
    if (!chatToUse) {
      try {
        const { data, error } = await supabase
          .from('chats')
          .insert({
            user_id: user.id,
            title: content.slice(0, 50) + (content.length > 50 ? '...' : ''),
            mode: 'models'
          })
          .select()
          .single();

        if (data && !error) {
          chatToUse = data;
          setChats(prev => [data, ...prev]);
          setCurrentChat(data);
        } else {
          return;
        }
      } catch (error) {
        console.error('Error creating chat:', error);
        return;
      }
    }

    // Ensure we have a valid chat before proceeding
    if (!chatToUse) {
      console.error('Failed to create or get chat');
      return;
    }

    setIsLoading(true);

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      await supabase.from('messages').insert({
        chat_id: chatToUse.id,
        role: 'user',
        content
      });

      const response = await fetch('/api/ai-workspace/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: chatToUse.id,
          prompt: content,
          mode: chatToUse.mode,
          userId: user.id
        })
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        created_at: new Date().toISOString(),
        tokens_used: data.tokens_used,
        eventId: data.eventId
      };
      setMessages(prev => [...prev, aiMessage]);

      await supabase.from('messages').insert({
        chat_id: chatToUse.id,
        role: 'assistant',
        content: data.response,
        tokens_used: data.tokens_used,
        model_used: data.model_used
      });

      if (data.eventId && data.status === 'processing') {
        setPendingModels(prev => new Set(prev).add(data.eventId));
        setTimeout(() => pollModelStatus(data.eventId, chatToUse.id), 5000);
      }

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

  const pollModelStatus = async (eventId: string, chatId: string) => {
    try {
      const response = await fetch(`/api/ai-workspace/status/${eventId}`);
      const data = await response.json();
      
      if (data.ready && data.model) {
        setPendingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(eventId);
          return newSet;
        });

        // Deploy to Hugging Face using user's token from env
        const deployResponse = await fetch('/api/ai-workspace/deploy-hf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            eventId,
            userId: user?.id
          })
        });

        const deployData = await deployResponse.json();
        
        if (deployData.success) {
          const completionMessage: Message = {
            id: `completion-${eventId}`,
            role: 'assistant',
            content: `# 🎉 **${data.model.name} - Ready & Deployed!**\n\nYour AI model has been successfully generated and deployed to Hugging Face!\n\n🔗 **Live Model URL:** [${deployData.spaceUrl || deployData.repoUrl}](${deployData.spaceUrl || deployData.repoUrl})\n\n## 📊 **Model Details**\n- **Name:** ${data.model.name}\n- **Type:** ${data.model.type.replace('-', ' ').toUpperCase()}\n- **Framework:** ${data.model.framework.toUpperCase()}\n- **Dataset:** ${data.model.dataset}\n- **Status:** ✅ Live on Hugging Face\n\n🚀 Your model is now accessible worldwide and ready for production use!`,
            created_at: new Date().toISOString(),
            eventId: eventId
          };
          
          setMessages(prev => [...prev, completionMessage]);

          if (supabase) {
            await supabase.from('messages').insert({
              chat_id: chatId,
              role: 'assistant',
              content: completionMessage.content,
              model_used: 'zehanx-ai-builder'
            });
          }
        }
      } else {
        setTimeout(() => pollModelStatus(eventId, chatId), 3000);
      }
    } catch (error) {
      console.error('Error polling model status:', error);
      setPendingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(eventId);
        return newSet;
      });
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      router.push('/login');
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading AI Workspace...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-gray-900 text-white flex flex-col overflow-hidden`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-700">
          <button
            onClick={createNewChat}
            className="w-full flex items-center justify-center space-x-2 border border-gray-600 text-white px-3 py-2 rounded-md hover:bg-gray-800 transition-colors text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New chat</span>
          </button>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto p-2">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer hover:bg-gray-800 ${
                currentChat?.id === chat.id ? 'bg-gray-800' : ''
              }`}
              onClick={() => {
                setCurrentChat(chat);
                loadMessages(chat.id);
              }}
            >
              <div className="flex items-center space-x-3 flex-1 min-w-0">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span className="text-sm truncate">{chat.title}</span>
              </div>
              <button
                onClick={(e) => deleteChat(chat.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-700 rounded transition-opacity"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>

        {/* User Section */}
        <div className="p-4 border-t border-gray-700">
          <div className="flex items-center space-x-3">
            <img
              src={user.user_metadata?.avatar_url || '/logo.jpg'}
              alt="User"
              className="w-8 h-8 rounded-full"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{user.email}</p>
              <p className="text-xs text-gray-400">AI Builder</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-lg font-semibold text-gray-900">
              {currentChat?.title || 'ChatGPT'}
            </h1>
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
          
          <div className="relative">
            <button
              onClick={() => setShowProfileMenu(!showProfileMenu)}
              className="flex items-center space-x-2 p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <img
                src={user.user_metadata?.avatar_url || '/logo.jpg'}
                alt="User"
                className="w-8 h-8 rounded-full"
              />
            </button>
            
            {showProfileMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 z-50">
                <div className="py-1">
                  <div className="px-4 py-2 text-sm text-gray-700 border-b border-gray-100">
                    <div className="font-medium">{user.email}</div>
                    <div className="text-gray-500">AI Builder</div>
                  </div>
                  <button
                    onClick={handleSignOut}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Sign out
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full px-4">
              <div className="text-center max-w-2xl">
                <h2 className="text-3xl font-semibold text-gray-900 mb-8">
                  What can I help with?
                </h2>
                
                <div className="grid grid-cols-2 gap-3 mb-8">
                  <button
                    onClick={() => setInputValue('Create a sentiment analysis model using BERT for analyzing customer reviews')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-green-600">🎯</span>
                      <span className="font-medium text-gray-900">Create image</span>
                    </div>
                    <p className="text-sm text-gray-600">Sentiment analysis with BERT</p>
                  </button>
                  
                  <button
                    onClick={() => setInputValue('Help me create an image classification model for detecting objects')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-blue-600">✏️</span>
                      <span className="font-medium text-gray-900">Help me write</span>
                    </div>
                    <p className="text-sm text-gray-600">Image classification model</p>
                  </button>
                  
                  <button
                    onClick={() => setInputValue('Summarize the best practices for training neural networks')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-yellow-600">📄</span>
                      <span className="font-medium text-gray-900">Summarize text</span>
                    </div>
                    <p className="text-sm text-gray-600">Neural network best practices</p>
                  </button>
                  
                  <button
                    onClick={() => setInputValue('Generate Python code for a chatbot using transformers')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-purple-600">💻</span>
                      <span className="font-medium text-gray-900">Code</span>
                    </div>
                    <p className="text-sm text-gray-600">Chatbot with transformers</p>
                  </button>
                  
                  <button
                    onClick={() => setInputValue('Brainstorm ideas for AI applications in healthcare')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-pink-600">💡</span>
                      <span className="font-medium text-gray-900">Brainstorm</span>
                    </div>
                    <p className="text-sm text-gray-600">AI in healthcare ideas</p>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6">
              {messages.map((message) => (
                <div key={message.id} className="mb-8">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      {message.role === 'user' ? (
                        <img
                          src={user.user_metadata?.avatar_url || '/logo.jpg'}
                          alt="User"
                          className="w-8 h-8 rounded-full"
                        />
                      ) : (
                        <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                          <span className="text-white text-sm font-bold">AI</span>
                        </div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 mb-1">
                        {message.role === 'user' ? 'You' : 'zehanx AI'}
                      </div>
                      <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap">
                        {message.content}
                      </div>
                      
                      {message.role === 'assistant' && (
                        <div className="mt-4 flex space-x-2">
                          <button className="text-gray-400 hover:text-gray-600 p-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L9 6v4m-5 8h2.5a2 2 0 002-2V8a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button className="text-gray-400 hover:text-gray-600 p-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 13l3 3 7-7" />
                            </svg>
                          </button>
                          <button className="text-gray-400 hover:text-gray-600 p-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button className="text-gray-400 hover:text-gray-600 p-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="mb-8">
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-bold">AI</span>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 mb-1">zehanx AI</div>
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600"></div>
                        <span className="text-gray-600">Generating your AI model...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {pendingModels.size > 0 && (
                <div className="mb-8">
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-bold">🔧</span>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 mb-1">System</div>
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                          <span className="text-blue-800">Building your AI model... This may take 1-2 minutes.</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (inputValue.trim() && !isLoading) {
                      sendMessage(inputValue.trim());
                      setInputValue('');
                    }
                  }
                }}
                placeholder="Message ChatGPT"
                className="w-full resize-none border border-gray-300 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
                style={{ minHeight: '52px', maxHeight: '200px' }}
              />
              <button
                onClick={() => {
                  if (inputValue.trim() && !isLoading) {
                    sendMessage(inputValue.trim());
                    setInputValue('');
                  }
                }}
                disabled={!inputValue.trim() || isLoading}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed rounded-lg transition-colors"
              >
                <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
            <div className="flex items-center justify-center mt-2 space-x-4 text-xs text-gray-500">
              <button className="hover:text-gray-700">📷 Create image</button>
              <button className="hover:text-gray-700">✏️ Help me write</button>
              <button className="hover:text-gray-700">📄 Summarize text</button>
              <button className="hover:text-gray-700">💻 Code</button>
              <button className="hover:text-gray-700">💡 Brainstorm</button>
            </div>
          </div>
        </div>
      </div>

      {/* Click outside to close profile menu */}
      {showProfileMenu && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setShowProfileMenu(false)}
        />
      )}
    </div>
  );
}