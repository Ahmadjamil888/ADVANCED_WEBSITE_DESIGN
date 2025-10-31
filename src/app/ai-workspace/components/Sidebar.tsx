"use client";

import React, { useState } from 'react';

interface Chat {
  id: string;
  title: string;
  mode: string;
  updated_at: string;
  is_pinned: boolean;
}

interface SidebarProps {
  chats: Chat[];
  currentChat: Chat | null;
  onChatSelect: (chat: Chat) => void;
  onNewChat: () => void;
  onSignOut: () => void;
}

export default function Sidebar({ chats, currentChat, onChatSelect, onNewChat, onSignOut }: SidebarProps) {
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) return 'Today';
    if (diffDays === 2) return 'Yesterday';
    if (diffDays <= 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'chat': return '💬';
      case 'code': return '💻';
      case 'research': return '🔬';
      case 'app-builder': return '🏗️';
      case 'translate': return '🌐';
      case 'fine-tune': return '⚙️';
      default: return '💬';
    }
  };

  const startEditing = (chat: Chat) => {
    setEditingChatId(chat.id);
    setEditTitle(chat.title);
  };

  const saveTitle = async (chatId: string) => {
    // TODO: Implement title update API call
    setEditingChatId(null);
  };

  return (
    <div className="w-full bg-gray-900 text-white flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <img src="/logo.jpg" alt="zehanxtech" className="w-6 h-6 rounded" />
            <span className="font-medium text-white text-sm">AI Workspace</span>
          </div>
        </div>
        
        {/* New Chat Button - ChatGPT Style */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center space-x-2 border border-gray-600 text-white px-3 py-2 rounded-md hover:bg-gray-800 transition-colors text-sm"
          aria-label="New conversation"
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
            onClick={() => onChatSelect(chat)}
            className={`group flex items-center space-x-3 p-2 rounded-md cursor-pointer transition-colors mb-1 ${
              currentChat?.id === chat.id 
                ? 'bg-gray-800 text-white' 
                : 'hover:bg-gray-800 text-gray-300'
            }`}
          >
            {/* Avatar/Icon */}
            <div className="flex-shrink-0">
              <div className="w-6 h-6 flex items-center justify-center text-xs">
                {getModeIcon(chat.mode)}
              </div>
            </div>

            {/* Chat Info */}
            <div className="flex-1 min-w-0">
              {editingChatId === chat.id ? (
                <input
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={() => saveTitle(chat.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveTitle(chat.id);
                    if (e.key === 'Escape') setEditingChatId(null);
                  }}
                  className="w-full text-sm text-white bg-gray-700 border border-gray-600 rounded px-2 py-1"
                  autoFocus
                />
              ) : (
                <div className="text-sm truncate">
                  {chat.title}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="relative">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    // TODO: Show context menu
                  }}
                  className="p-1 text-gray-400 hover:text-gray-600 rounded"
                  aria-label="Chat options"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Pin indicator */}
            {chat.is_pinned && (
              <div className="flex-shrink-0">
                <svg className="w-3 h-3 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-700">
        <button
          onClick={onSignOut}
          className="w-full flex items-center space-x-2 text-gray-300 hover:text-white px-3 py-2 rounded-md hover:bg-gray-800 transition-colors text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
          <span>Sign out</span>
        </button>
      </div>
    </div>
  );
}