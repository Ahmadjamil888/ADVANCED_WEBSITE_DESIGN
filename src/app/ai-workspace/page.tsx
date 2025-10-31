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

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
      return;
    }

    if (user && supabase) {
      loadChats();
    }
  }, [user, loading, router]);

  const loadChats = async () => {
    if (!supabase || !user) return;

    try {
      const { data, error } = await supabase
        .from('chats')
        .select('*')
        .eq('user_id', user.id)
        .order('updated_at', { ascending: false });

      if (data && !error) {
        setChats(data);
        // Don't auto-select first chat - start with empty state
      }
    } catch (err) {
      console.error('Error loading chats:', err);
    }
  };

  const loadMessages = async (chatId: string) => {
    if (!supabase) return;

    try {
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .eq('chat_id', chatId)
        .order('created_at', { ascending: true });

      if (data && !error) {
        setMessages(data);
      }
    } catch (err) {
      console.error('Error loading messages:', err);
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
        setInputValue('');
      } else {
        console.error('Error creating chat:', error);
      }
    } catch (error) {
      console.error('Error in createNewChat:', error);
    }
  };

  const deleteChat = async (chatId: string, e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    
    if (!supabase || !user) return;

    try {
      // Delete messages first
      await supabase
        .from('messages')
        .delete()
        .eq('chat_id', chatId);

      // Delete chat
      const { error: chatError } = await supabase
        .from('chats')
        .delete()
        .eq('id', chatId)
        .eq('user_id', user.id);

      if (!chatError) {
        // Update UI
        setChats(prev => prev.filter(chat => chat.id !== chatId));
        
        // If this was the current chat, clear it
        if (currentChat?.id === chatId) {
          setCurrentChat(null);
          setMessages([]);
        }
      }
    } catch (error) {
      console.error('Error in deleteChat:', error);
    }
  };

  const handleExampleClick = (exampleText: string) => {
    setInputValue(exampleText);
    // Auto-focus the textarea
    setTimeout(() => {
      const textarea = document.querySelector('textarea');
      if (textarea) {
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      }
    }, 100);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInputValue(value);
    
    // Auto-resize textarea
    const target = e.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !isLoading) {
        sendMessage(inputValue.trim());
      }
    }
  };

  const handleSendClick = () => {
    if (inputValue.trim() && !isLoading) {
      sendMessage(inputValue.trim());
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      console.log('Copied to clipboard');
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const regenerateResponse = async (messageIndex: number) => {
    if (!currentChat || isLoading) return;
    
    const userMessage = messages[messageIndex - 1];
    if (userMessage && userMessage.role === 'user') {
      await sendMessage(userMessage.content);
    }
  };

  const rateResponse = async (messageId: string, rating: 'good' | 'bad') => {
    console.log(`Rated message ${messageId} as ${rating}`);
  };

  const sendMessage = async (content: string) => {
    if (!supabase || !user || isLoading || !content.trim()) return;

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
          console.error('Error creating chat for message:', error);
          return;
        }
      } catch (error) {
        console.error('Error creating new chat:', error);
        return;
      }
    }

    // At this point, chatToUse is guaranteed to be non-null
    const activeChat = chatToUse as Chat;

    setIsLoading(true);
    setInputValue('');
    
    // Reset textarea height
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'auto';
    }

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
        chat_id: activeChat.id,
        role: 'user',
        content
      });

      // Generate AI response
      const response = await fetch('/api/ai-workspace/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId: activeChat.id,
          prompt: content,
          mode: activeChat.mode,
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
        tokens_used: data.tokens_used,
        eventId: data.eventId
      };
      setMessages(prev => [...prev, aiMessage]);

      // Save AI message to database
      await supabase.from('messages').insert({
        chat_id: activeChat.id,
        role: 'assistant',
        content: data.response,
        tokens_used: data.tokens_used,
        model_used: data.model_used
      });

      // Update chat timestamp
      await supabase
        .from('chats')
        .update({ updated_at: new Date().toISOString() })
        .eq('id', activeChat.id);

      // Start polling for model completion if this is a model generation request
      if (data.eventId && data.status === 'processing') {
        setPendingModels(prev => new Set(prev).add(data.eventId));
        setTimeout(() => pollModelStatus(data.eventId), 5000);
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

  const pollModelStatus = async (eventId: string) => {
    try {
      const response = await fetch(`/api/ai-workspace/status/${eventId}`);
      const data = await response.json();
      
      if (data.ready && data.model) {
        setPendingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(eventId);
          return newSet;
        });

        // Auto-deploy to Hugging Face using env token
        try {
          const deployResponse = await fetch('/api/ai-workspace/deploy-hf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              eventId,
              userId: user?.id,
              autoUseEnvToken: true
            })
          });

          const deployData = await deployResponse.json();
          
          if (deployData.success) {
            const completionMessage: Message = {
              id: `completion-${eventId}`,
              role: 'assistant',
              content: `🎉 **${data.model.name} - Ready & Deployed!**

Your AI model has been successfully generated and deployed to Hugging Face!

🔗 **Live Model URL:** [${deployData.repoUrl}](${deployData.repoUrl})

**Model Details:**
- **Name:** ${data.model.name}
- **Type:** ${data.model.type.replace('-', ' ').toUpperCase()}
- **Framework:** ${data.model.framework.toUpperCase()}
- **Dataset:** ${data.model.dataset}
- **Status:** ✅ Live on Hugging Face

Your model is now accessible worldwide and ready for production use!`,
              created_at: new Date().toISOString(),
              eventId: eventId
            };
            
            setMessages(prev => [...prev, completionMessage]);

            if (supabase && currentChat) {
              await supabase.from('messages').insert({
                chat_id: currentChat.id,
                role: 'assistant',
                content: completionMessage.content,
                model_used: 'zehanx-ai-builder'
              });
            }
          }
        } catch (deployError) {
          console.error('Deployment error:', deployError);
        }
      } else {
        setTimeout(() => pollModelStatus(eventId), 3000);
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
      <div className="flex items-center justify-center h-screen bg-white">
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

  if (!supabase) {
    return (
      <div className="flex items-center justify-center h-screen bg-white">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Database Connection Error</h3>
          <p className="text-gray-600">Unable to connect to the database. Please try refreshing the page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-gray-900 text-white flex flex-col overflow-hidden`}>
        {/* Sidebar Header */}
        <div className="p-3 border-b border-gray-700">
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
          {chats.length === 0 ? (
            <div className="text-center text-gray-400 text-sm mt-8">
              No chats yet
            </div>
          ) : (
            chats.map((chat) => (
              <div
                key={chat.id}
                className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer hover:bg-gray-800 mb-1 ${
                  currentChat?.id === chat.id ? 'bg-gray-800' : ''
                }`}
                onClick={() => {
                  setCurrentChat(chat);
                  loadMessages(chat.id);
                }}
              >
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <span className="text-sm truncate">{chat.title}</span>
                </div>
                <button
                  onClick={(e) => deleteChat(chat.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-700 rounded transition-all flex-shrink-0"
                  title="Delete chat"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))
          )}
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
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-lg font-semibold text-gray-900">
              {currentChat?.title || 'zehanx AI'}
            </h1>
          </div>
          
          {/* Sign Out Button - Top Right */}
          <button
            onClick={handleSignOut}
            className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <img
              src={user.user_metadata?.avatar_url || '/logo.jpg'}
              alt="User"
              className="w-6 h-6 rounded-full"
            />
            <span>Sign out</span>
          </button>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            /* Empty State - Exact ChatGPT Design */
            <div className="flex flex-col items-center justify-center h-full px-4">
              <div className="text-center max-w-2xl">
                <h2 className="text-3xl font-semibold text-gray-900 mb-8">
                  What can I help with?
                </h2>
                
                {/* Action Cards Grid */}
                <div className="grid grid-cols-2 gap-3 mb-8 max-w-2xl">
                  <button
                    onClick={() => handleExampleClick('Create a sentiment analysis model using BERT for analyzing customer reviews and feedback')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors group"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-green-600">🎯</span>
                      <span className="font-medium text-gray-900">Create model</span>
                    </div>
                    <p className="text-sm text-gray-600">Sentiment analysis with BERT</p>
                  </button>
                  
                  <button
                    onClick={() => handleExampleClick('Help me create an image classification model using ResNet for detecting objects in photos')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors group"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-blue-600">✏️</span>
                      <span className="font-medium text-gray-900">Help me write</span>
                    </div>
                    <p className="text-sm text-gray-600">Image classification model</p>
                  </button>
                  
                  <button
                    onClick={() => handleExampleClick('Summarize the best practices for training deep neural networks and optimizing model performance')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors group"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-yellow-600">📄</span>
                      <span className="font-medium text-gray-900">Summarize text</span>
                    </div>
                    <p className="text-sm text-gray-600">Neural network best practices</p>
                  </button>
                  
                  <button
                    onClick={() => handleExampleClick('Generate Python code for a conversational AI chatbot using transformers and Hugging Face')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors group"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-purple-600">💻</span>
                      <span className="font-medium text-gray-900">Code</span>
                    </div>
                    <p className="text-sm text-gray-600">Chatbot with transformers</p>
                  </button>
                  
                  <button
                    onClick={() => handleExampleClick('Brainstorm innovative AI applications for healthcare, finance, and education sectors')}
                    className="p-4 text-left bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-colors group"
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-pink-600">💡</span>
                      <span className="font-medium text-gray-900">Brainstorm</span>
                    </div>
                    <p className="text-sm text-gray-600">AI applications ideas</p>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Chat Messages - Exact ChatGPT Design */
            <div className="max-w-3xl mx-auto px-4 py-6">
              {messages.map((message, index) => (
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
                      
                      {/* Action Buttons for AI Messages */}
                      {message.role === 'assistant' && (
                        <div className="mt-4 flex space-x-2">
                          <button 
                            onClick={() => rateResponse(message.id, 'good')}
                            className="text-gray-400 hover:text-green-600 p-1 rounded hover:bg-gray-100 transition-colors" 
                            title="Good response"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L9 6v4m-5 8h2.5a2 2 0 002-2V8a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button 
                            onClick={() => rateResponse(message.id, 'bad')}
                            className="text-gray-400 hover:text-red-600 p-1 rounded hover:bg-gray-100 transition-colors" 
                            title="Bad response"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v2a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L15 18v-4m-5-8h2.5a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V8a2 2 0 012-2z" />
                            </svg>
                          </button>
                          <button 
                            onClick={() => copyToClipboard(message.content)}
                            className="text-gray-400 hover:text-blue-600 p-1 rounded hover:bg-gray-100 transition-colors" 
                            title="Copy"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button 
                            onClick={() => regenerateResponse(index)}
                            className="text-gray-400 hover:text-purple-600 p-1 rounded hover:bg-gray-100 transition-colors" 
                            title="Regenerate"
                            disabled={isLoading}
                          >
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
              
              {/* Loading State */}
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
              
              {/* Pending Models Status */}
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

        {/* Input Section - Exact ChatGPT Design */}
        <div className="border-t border-gray-200 p-4 bg-white">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                value={inputValue}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Message zehanx AI..."
                className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-500"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '200px' }}
                disabled={isLoading}
              />
              <button
                onClick={handleSendClick}
                disabled={!inputValue.trim() || isLoading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
            
            {/* Footer Text */}
            <p className="text-xs text-gray-500 text-center mt-3">
              zehanx AI can generate, train, and deploy custom AI models. Always verify generated code before training.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}