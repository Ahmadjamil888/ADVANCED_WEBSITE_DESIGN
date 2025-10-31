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
      await supabase.from('messages').delete().eq('chat_id', chatId);
      const { error: chatError } = await supabase
        .from('chats')
        .delete()
        .eq('id', chatId)
        .eq('user_id', user.id);

      if (!chatError) {
        setChats(prev => prev.filter(chat => chat.id !== chatId));
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
    setTimeout(() => {
      const textarea = document.querySelector('textarea');
      if (textarea) {
        textarea.focus();
      }
    }, 100);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInputValue(value);
    
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
      alert('Copied to clipboard!');
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
    alert(`Rated as ${rating}!`);
  };

  const sendMessage = async (content: string) => {
    if (!supabase || !user || isLoading || !content.trim()) return;

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

    const activeChat = chatToUse as Chat;

    setIsLoading(true);
    setInputValue('');
    
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'auto';
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      await supabase.from('messages').insert({
        chat_id: activeChat.id,
        role: 'user',
        content
      });

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
        chat_id: activeChat.id,
        role: 'assistant',
        content: data.response,
        tokens_used: data.tokens_used,
        model_used: data.model_used
      });

      await supabase
        .from('chats')
        .update({ updated_at: new Date().toISOString() })
        .eq('id', activeChat.id);

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
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100vh', 
        backgroundColor: 'white' 
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            width: '32px', 
            height: '32px', 
            border: '2px solid #e5e7eb', 
            borderTop: '2px solid #2563eb', 
            borderRadius: '50%', 
            animation: 'spin 1s linear infinite',
            margin: '0 auto 16px'
          }}></div>
          <p style={{ color: '#6b7280' }}>Loading AI Workspace...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  if (!supabase) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100vh', 
        backgroundColor: 'white' 
      }}>
        <div style={{ textAlign: 'center', maxWidth: '400px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#111827', marginBottom: '8px' }}>
            Database Connection Error
          </h3>
          <p style={{ color: '#6b7280' }}>Unable to connect to the database. Please try refreshing the page.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <style jsx global>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        html, body {
          height: 100%;
          overflow: hidden;
        }
        #__next {
          height: 100%;
        }
      `}</style>
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .sidebar {
          width: ${sidebarOpen ? '256px' : '0px'};
          transition: width 0.3s ease;
          background-color: #111827;
          color: white;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          height: 100vh;
        }
        .sidebar-header {
          padding: 12px;
          border-bottom: 1px solid #374151;
        }
        .new-chat-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          border: 1px solid #4b5563;
          color: white;
          padding: 8px 12px;
          border-radius: 6px;
          background: transparent;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.2s;
        }
        .new-chat-btn:hover {
          background-color: #1f2937;
        }
        .chat-list {
          flex: 1;
          overflow-y: auto;
          padding: 8px;
        }
        .chat-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px;
          border-radius: 8px;
          cursor: pointer;
          margin-bottom: 4px;
          transition: background-color 0.2s;
        }
        .chat-item:hover {
          background-color: #1f2937;
        }
        .chat-item.active {
          background-color: #1f2937;
        }
        .chat-content {
          display: flex;
          align-items: center;
          gap: 12px;
          flex: 1;
          min-width: 0;
        }
        .chat-title {
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .delete-btn {
          opacity: 0;
          padding: 4px;
          border-radius: 4px;
          background: transparent;
          border: none;
          color: white;
          cursor: pointer;
          transition: all 0.2s;
        }
        .chat-item:hover .delete-btn {
          opacity: 1;
        }
        .delete-btn:hover {
          background-color: #374151;
        }
        .user-section {
          padding: 16px;
          border-top: 1px solid #374151;
        }
        .user-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .user-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
        }
        .user-details {
          flex: 1;
          min-width: 0;
        }
        .user-email {
          font-size: 14px;
          font-weight: 500;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .user-role {
          font-size: 12px;
          color: #9ca3af;
        }
        .main-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100vh;
          overflow: hidden;
        }
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px;
          border-bottom: 1px solid #e5e7eb;
          background: white;
          min-height: 70px;
          flex-shrink: 0;
        }
        .header-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .sidebar-toggle {
          padding: 8px;
          border-radius: 8px;
          background: transparent;
          border: none;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        .sidebar-toggle:hover {
          background-color: #f3f4f6;
        }
        .header-title {
          font-size: 18px;
          font-weight: 600;
          color: #111827;
        }
        .signout-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          font-size: 14px;
          color: #6b7280;
          background: transparent;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
          min-width: fit-content;
        }
        .signout-btn:hover {
          color: #111827;
          background-color: #f3f4f6;
        }
        .signout-avatar {
          width: 24px;
          height: 24px;
          border-radius: 50%;
        }
        .messages-area {
          flex: 1;
          overflow-y: auto;
          height: calc(100vh - 140px);
        }
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          padding: 16px;
        }
        .empty-title {
          font-size: 30px;
          font-weight: 600;
          color: #111827;
          margin-bottom: 32px;
          text-align: center;
        }
        .example-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
          margin-bottom: 32px;
          max-width: 600px;
        }
        .example-card {
          padding: 16px;
          text-align: left;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        .example-card:hover {
          background: #f3f4f6;
        }
        .example-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }
        .example-title {
          font-weight: 500;
          color: #111827;
        }
        .example-desc {
          font-size: 14px;
          color: #6b7280;
        }
        .messages-container {
          max-width: 768px;
          margin: 0 auto;
          padding: 24px 16px;
        }
        .message {
          margin-bottom: 32px;
        }
        .message-content {
          display: flex;
          align-items: flex-start;
          gap: 16px;
        }
        .message-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .ai-avatar {
          width: 32px;
          height: 32px;
          background: #059669;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 14px;
          font-weight: bold;
        }
        .message-body {
          flex: 1;
          min-width: 0;
        }
        .message-sender {
          font-size: 14px;
          font-weight: 500;
          color: #111827;
          margin-bottom: 4px;
        }
        .message-text {
          color: #1f2937;
          white-space: pre-wrap;
          line-height: 1.6;
        }
        .message-actions {
          margin-top: 16px;
          display: flex;
          gap: 8px;
        }
        .action-btn {
          padding: 4px;
          border-radius: 4px;
          background: transparent;
          border: none;
          color: #9ca3af;
          cursor: pointer;
          transition: color 0.2s;
        }
        .action-btn:hover {
          background: #f3f4f6;
        }
        .action-btn:hover.good { color: #059669; }
        .action-btn:hover.bad { color: #dc2626; }
        .action-btn:hover.copy { color: #2563eb; }
        .action-btn:hover.regen { color: #7c3aed; }
        .loading-message {
          margin-bottom: 32px;
        }
        .loading-content {
          display: flex;
          align-items: flex-start;
          gap: 16px;
        }
        .loading-body {
          flex: 1;
        }
        .loading-text {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid #e5e7eb;
          border-top: 2px solid #059669;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        .input-section {
          border-top: 1px solid #e5e7eb;
          padding: 16px;
          background: white;
          flex-shrink: 0;
          min-height: 70px;
        }
        .input-container {
          max-width: 768px;
          margin: 0 auto;
        }
        .input-wrapper {
          position: relative;
        }
        .input-textarea {
          width: 100%;
          padding: 12px 48px 12px 12px;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          resize: none;
          outline: none;
          color: #111827;
          min-height: 44px;
          max-height: 200px;
          font-family: inherit;
          font-size: 14px;
        }
        .input-textarea:focus {
          border-color: #2563eb;
          box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        .input-textarea::placeholder {
          color: #9ca3af;
        }
        .send-btn {
          position: absolute;
          right: 8px;
          top: 50%;
          transform: translateY(-50%);
          padding: 8px;
          background: transparent;
          border: none;
          color: #9ca3af;
          cursor: pointer;
          transition: color 0.2s;
        }
        .send-btn:hover:not(:disabled) {
          color: #6b7280;
        }
        .send-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .footer-text {
          font-size: 12px;
          color: #9ca3af;
          text-align: center;
          margin-top: 12px;
        }
      `}</style>
      
      <div style={{ 
        display: 'flex', 
        height: '100vh', 
        backgroundColor: 'white',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden'
      }}>
        {/* Sidebar */}
        <div className="sidebar">
          {/* Sidebar Header */}
          <div className="sidebar-header">
            <button onClick={createNewChat} className="new-chat-btn">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span>New chat</span>
            </button>
          </div>

          {/* Chat List */}
          <div className="chat-list">
            {chats.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#9ca3af', fontSize: '14px', marginTop: '32px' }}>
                No chats yet
              </div>
            ) : (
              chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`chat-item ${currentChat?.id === chat.id ? 'active' : ''}`}
                  onClick={() => {
                    setCurrentChat(chat);
                    loadMessages(chat.id);
                  }}
                >
                  <div className="chat-content">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <span className="chat-title">{chat.title}</span>
                  </div>
                  <button
                    onClick={(e) => deleteChat(chat.id, e)}
                    className="delete-btn"
                    title="Delete chat"
                  >
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ))
            )}
          </div>

          {/* User Section */}
          <div className="user-section">
            <div className="user-info">
              <img
                src={user.user_metadata?.avatar_url || '/logo.jpg'}
                alt="User"
                className="user-avatar"
              />
              <div className="user-details">
                <p className="user-email">{user.email}</p>
                <p className="user-role">AI Builder</p>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Top Header */}
          <div className="header">
            <div className="header-left">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="sidebar-toggle"
              >
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <h1 className="header-title">
                {currentChat?.title || 'zehanx AI'}
              </h1>
            </div>
            
            {/* Sign Out Button - Top Right */}
            <button onClick={handleSignOut} className="signout-btn">
              <img
                src={user.user_metadata?.avatar_url || '/logo.jpg'}
                alt="User"
                className="signout-avatar"
              />
              <span>Sign out</span>
            </button>
          </div>

          {/* Chat Messages Area */}
          <div className="messages-area">
            {messages.length === 0 ? (
              /* Empty State */
              <div className="empty-state">
                <div style={{ textAlign: 'center', maxWidth: '600px' }}>
                  <h2 className="empty-title">What can I help with?</h2>
                  
                  {/* Action Cards Grid */}
                  <div className="example-grid">
                    <button
                      onClick={() => handleExampleClick('Create a sentiment analysis model using BERT for analyzing customer reviews and feedback')}
                      className="example-card"
                    >
                      <div className="example-header">
                        <span style={{ color: '#059669' }}>🎯</span>
                        <span className="example-title">Create model</span>
                      </div>
                      <p className="example-desc">Sentiment analysis with BERT</p>
                    </button>
                    
                    <button
                      onClick={() => handleExampleClick('Help me create an image classification model using ResNet for detecting objects in photos')}
                      className="example-card"
                    >
                      <div className="example-header">
                        <span style={{ color: '#2563eb' }}>✏️</span>
                        <span className="example-title">Help me write</span>
                      </div>
                      <p className="example-desc">Image classification model</p>
                    </button>
                    
                    <button
                      onClick={() => handleExampleClick('Summarize the best practices for training deep neural networks and optimizing model performance')}
                      className="example-card"
                    >
                      <div className="example-header">
                        <span style={{ color: '#d97706' }}>📄</span>
                        <span className="example-title">Summarize text</span>
                      </div>
                      <p className="example-desc">Neural network best practices</p>
                    </button>
                    
                    <button
                      onClick={() => handleExampleClick('Generate Python code for a conversational AI chatbot using transformers and Hugging Face')}
                      className="example-card"
                    >
                      <div className="example-header">
                        <span style={{ color: '#7c3aed' }}>💻</span>
                        <span className="example-title">Code</span>
                      </div>
                      <p className="example-desc">Chatbot with transformers</p>
                    </button>
                    
                    <button
                      onClick={() => handleExampleClick('Brainstorm innovative AI applications for healthcare, finance, and education sectors')}
                      className="example-card"
                    >
                      <div className="example-header">
                        <span style={{ color: '#ec4899' }}>💡</span>
                        <span className="example-title">Brainstorm</span>
                      </div>
                      <p className="example-desc">AI applications ideas</p>
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              /* Chat Messages */
              <div className="messages-container">
                {messages.map((message, index) => (
                  <div key={message.id} className="message">
                    <div className="message-content">
                      <div>
                        {message.role === 'user' ? (
                          <img
                            src={user.user_metadata?.avatar_url || '/logo.jpg'}
                            alt="User"
                            className="message-avatar"
                          />
                        ) : (
                          <div className="ai-avatar">AI</div>
                        )}
                      </div>
                      <div className="message-body">
                        <div className="message-sender">
                          {message.role === 'user' ? 'You' : 'zehanx AI'}
                        </div>
                        <div className="message-text">{message.content}</div>
                        
                        {/* Action Buttons for AI Messages */}
                        {message.role === 'assistant' && (
                          <div className="message-actions">
                            <button 
                              onClick={() => rateResponse(message.id, 'good')}
                              className="action-btn good" 
                              title="Good response"
                            >
                              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L9 6v4m-5 8h2.5a2 2 0 002-2V8a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2z" />
                              </svg>
                            </button>
                            <button 
                              onClick={() => rateResponse(message.id, 'bad')}
                              className="action-btn bad" 
                              title="Bad response"
                            >
                              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v2a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L15 18v-4m-5-8h2.5a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V8a2 2 0 012-2z" />
                              </svg>
                            </button>
                            <button 
                              onClick={() => copyToClipboard(message.content)}
                              className="action-btn copy" 
                              title="Copy"
                            >
                              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                              </svg>
                            </button>
                            <button 
                              onClick={() => regenerateResponse(index)}
                              className="action-btn regen" 
                              title="Regenerate"
                              disabled={isLoading}
                            >
                              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                  <div className="loading-message">
                    <div className="loading-content">
                      <div className="ai-avatar">AI</div>
                      <div className="loading-body">
                        <div className="message-sender">zehanx AI</div>
                        <div className="loading-text">
                          <div className="spinner"></div>
                          <span style={{ color: '#6b7280' }}>Generating your AI model...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Pending Models Status */}
                {pendingModels.size > 0 && (
                  <div className="loading-message">
                    <div className="loading-content">
                      <div style={{ 
                        width: '32px', 
                        height: '32px', 
                        background: '#2563eb', 
                        borderRadius: '50%', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center', 
                        color: 'white', 
                        fontSize: '14px', 
                        fontWeight: 'bold' 
                      }}>🔧</div>
                      <div className="loading-body">
                        <div className="message-sender">System</div>
                        <div style={{ 
                          background: '#eff6ff', 
                          border: '1px solid #bfdbfe', 
                          borderRadius: '8px', 
                          padding: '12px' 
                        }}>
                          <div className="loading-text">
                            <div className="spinner"></div>
                            <span style={{ color: '#1e40af' }}>Building your AI model... This may take 1-2 minutes.</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input Section */}
          <div className="input-section">
            <div className="input-container">
              <div className="input-wrapper">
                <textarea
                  value={inputValue}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  placeholder="Message zehanx AI..."
                  className="input-textarea"
                  rows={1}
                  disabled={isLoading}
                />
                <button
                  onClick={handleSendClick}
                  disabled={!inputValue.trim() || isLoading}
                  className="send-btn"
                >
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
              
              {/* Footer Text */}
              <p className="footer-text">
                zehanx AI can generate, train, and deploy custom AI models. Always verify generated code before training.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}