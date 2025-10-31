"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

// Import components
import Sidebar from "./components/Sidebar";
import MainContent from "./components/MainContent";
import ContextPanel from "./components/ContextPanel";

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
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        currentChat={currentChat}
        onChatSelect={(chat) => {
          setCurrentChat(chat);
          loadMessages(chat.id);
        }}
        onNewChat={createNewChat}
        onSignOut={signOut}
      />

      {/* Main Content */}
      <div className="flex-1 flex">
        <MainContent
          currentChat={currentChat}
          messages={messages}
          currentMode={currentMode}
          onModeChange={setCurrentMode}
          onSendMessage={sendMessage}
          isLoading={isLoading}
          contextPanelOpen={contextPanelOpen}
          onToggleContextPanel={() => setContextPanelOpen(!contextPanelOpen)}
        />

        {/* Context Panel */}
        {contextPanelOpen && (
          <ContextPanel
            currentChat={currentChat}
            onClose={() => setContextPanelOpen(false)}
          />
        )}
      </div>
    </div>
  );
}