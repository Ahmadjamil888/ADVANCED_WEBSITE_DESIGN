"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter, useSearchParams } from "next/navigation";
import { supabase } from "@/lib/supabase";


interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const aiModels = [
  {
    id: "assistant",
    name: "AI Assistant",
    description: "General purpose AI assistant powered by zehanx AI",
    icon: "🤖",
    category: "General"
  },
  {
    id: "quiz",
    name: "AI Quiz Generator", 
    description: "Generate quizzes and educational content",
    icon: "📝",
    category: "Education"
  },
  {
    id: "helper",
    name: "AI Helper",
    description: "Get help with various tasks and questions",
    icon: "🆘",
    category: "Utility"
  },
  {
    id: "image-analyzer",
    name: "AI Image Analyzer",
    description: "Analyze and describe images using AI",
    icon: "🖼️",
    category: "Vision"
  },
  {
    id: "researcher",
    name: "AI Researcher",
    description: "Research topics and provide detailed insights",
    icon: "🔬",
    category: "Research"
  },
  {
    id: "doc-maker",
    name: "AI Doc Maker",
    description: "Create and format documents automatically",
    icon: "📄",
    category: "Productivity"
  }
];

export default function Dashboard() {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [userStats, setUserStats] = useState({
    totalRequests: 0,
    dailyRequests: 0,
    monthlyRequests: 0,
    isPremium: false
  });

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
      return;
    }

    if (user) {
      // Redirect to AI workspace instead of showing dashboard
      router.push("/ai-workspace");
      return;
    }
  }, [user, loading, router]);

  const fetchUserStats = async () => {
    if (!user || !supabase) return;

    const { data } = await supabase
      .from("users")
      .select("total_requests, daily_requests, monthly_requests, is_premium")
      .eq("id", user.id)
      .single();

    if (data) {
      setUserStats({
        totalRequests: data.total_requests || 0,
        dailyRequests: data.daily_requests || 0,
        monthlyRequests: data.monthly_requests || 0,
        isPremium: data.is_premium || false
      });
    }
  };

  const handleSendMessage = async (message?: string, model?: string) => {
    const messageText = message || inputMessage;
    const currentModel = model || selectedModel;
    
    if (!messageText.trim() || !currentModel || isLoading) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: messageText,
          model: currentModel,
          userId: user?.id
        }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiResponse]);

      if (supabase) {
        await supabase
          .from("users")
          .update({
            total_requests: userStats.totalRequests + 1,
            daily_requests: userStats.dailyRequests + 1,
            monthly_requests: userStats.monthlyRequests + 1,
            last_activity: new Date().toISOString()
          })
          .eq("id", user?.id);

        await supabase.from("model_usage").insert({
          user_id: user?.id,
          model_name: currentModel,
          prompt_tokens: messageText.length,
          completion_tokens: data.response.length,
          total_tokens: messageText.length + data.response.length,
          cost_credits: 1
        });

        fetchUserStats();
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Error: ${error.message}`,
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-text">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  // If no model selected, show dashboard overview
  if (!selectedModel) {
    return (
      <div className="dashboard-container">
        {/* Sidebar */}
        <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
          <div className="sidebar-header">
            <div className="sidebar-brand">
              <img src="/logo.jpg" alt="zehanx AI" className="sidebar-logo" />
              {sidebarOpen && (
                <div>
                  <h2 className="sidebar-title">zehanx AI</h2>
                  <p className="sidebar-subtitle">Dashboard</p>
                </div>
              )}
            </div>
          </div>
          
          <div className="sidebar-menu">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="menu-button"
            >
              <svg className="menu-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              {sidebarOpen && <span>Menu</span>}
            </button>
          </div>

          <div style={{ padding: '0 1rem 1rem 1rem' }}>
            <button
              onClick={signOut}
              className="sign-out-button"
            >
              <svg className="menu-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              {sidebarOpen && <span>Sign Out</span>}
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          <div className="dashboard-content">
            {/* Header */}
            <div className="dashboard-header">
              <h1 className="dashboard-title">Welcome to zehanx AI</h1>
              <p className="dashboard-subtitle">Choose an AI model to get started</p>
            </div>

            {/* Stats Cards */}
            <div className="stats-grid">
              <div className="stat-card">
                <h3 className="stat-title">Total Requests</h3>
                <p className="stat-value">{userStats.totalRequests}</p>
              </div>
              <div className="stat-card">
                <h3 className="stat-title">Daily Requests</h3>
                <p className="stat-value">{userStats.dailyRequests}</p>
              </div>
              <div className="stat-card">
                <h3 className="stat-title">Monthly Requests</h3>
                <p className="stat-value">{userStats.monthlyRequests}</p>
              </div>
              <div className="stat-card">
                <h3 className="stat-title">Plan</h3>
                <p className="stat-value">{userStats.isPremium ? "Pro" : "Free"}</p>
              </div>
            </div>

            {/* AI Models Grid */}
            <div className="models-grid">
              {aiModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className="model-card"
                >
                  <div className="model-header">
                    <span className="model-icon">{model.icon}</span>
                    <div className="model-info">
                      <h3>{model.name}</h3>
                      <span className="model-category">
                        {model.category}
                      </span>
                    </div>
                  </div>
                  <p className="model-description">{model.description}</p>
                  <div className="model-footer">
                    Powered by zehanx AI
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Chat interface for selected model
  const currentModel = aiModels.find(m => m.id === selectedModel);

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-content">
          <div className="chat-nav">
            <button
              onClick={() => setSelectedModel(null)}
              className="back-button"
            >
              ← Back to Dashboard
            </button>
            <span className="chat-title">{currentModel?.name}</span>
          </div>
          <button
            onClick={signOut}
            className="back-button"
            style={{ color: '#dc2626' }}
          >
            Sign Out
          </button>
        </div>
      </div>

        {messages.length === 0 ? (
          /* Welcome Screen */
          <div className="welcome-screen">
            <div className="welcome-content">
              <img
                className="welcome-logo"
                src="/logo.jpg"
                alt="zehanx AI"
              />
              <div className="suggestions-grid">
                <div 
                  className="suggestion-card"
                  onClick={() => setInputMessage("Create an image for my slide deck")}
                >
                  <svg className="suggestion-icon" style={{ color: '#3b82f6' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="suggestion-text">
                    Create an image for my slide deck
                  </span>
                </div>
                <div 
                  className="suggestion-card"
                  onClick={() => setInputMessage("Thank my interviewer")}
                >
                  <svg className="suggestion-icon" style={{ color: '#ef4444' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  <span className="suggestion-text">
                    Thank my interviewer
                  </span>
                </div>
                <div 
                  className="suggestion-card"
                  onClick={() => setInputMessage("Plan a relaxing day")}
                >
                  <svg className="suggestion-icon" style={{ color: '#eab308' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                  <span className="suggestion-text">
                    Plan a relaxing day
                  </span>
                </div>
                <div 
                  className="suggestion-card"
                  onClick={() => setInputMessage("Explain nostalgia like I'm 5")}
                >
                  <svg className="suggestion-icon" style={{ color: '#22c55e' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
                  </svg>
                  <span className="suggestion-text">
                    Explain nostalgia like I'm 5
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Chat Messages */
          <div className="chat-messages">
            <div className="messages-container">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.isUser ? "user" : "ai"}`}
                >
                  <div
                    className={`message-bubble ${message.isUser ? "user" : "ai"}`}
                  >
                    <p className="message-text">{message.text}</p>
                    <p className="message-time">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="typing-indicator">
                  <div className="typing-bubble">
                    <div className="typing-dots">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <span className="typing-text">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="chat-input-area">
          <div className="chat-input-container">
            <button className="input-button">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.586-6.586a2 2 0 00-2.828-2.828z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4" />
              </svg>
            </button>
            <input
              className="chat-input"
              placeholder="Ask me anything"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button 
              disabled={!inputMessage.trim() || isLoading}
              onClick={() => handleSendMessage()}
              className="send-button"
            >
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </button>
          </div>
          <span className="chat-disclaimer">
            zehanx AI can make mistakes. Check important info.
          </span>
        </div>
    </div>
  );
}