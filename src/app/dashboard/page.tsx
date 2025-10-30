"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter, useSearchParams } from "next/navigation";
import { supabase } from "@/lib/supabase";
import Script from "next/script";

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
      fetchUserStats();
      const model = searchParams.get("model");
      const prompt = searchParams.get("prompt");
      
      if (model) {
        setSelectedModel(model);
        if (prompt) {
          setInputMessage(prompt);
          handleSendMessage(prompt, model);
        }
      }
    }
  }, [user, loading, router, searchParams]);

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
      <>
        <Script src="https://cdn.tailwindcss.com" />
        <div className="flex h-screen w-full items-center justify-center bg-white">
          <div className="text-lg text-gray-900">Loading...</div>
        </div>
      </>
    );
  }

  if (!user) {
    return null;
  }

  // If no model selected, show dashboard overview
  if (!selectedModel) {
    return (
      <>
        <Script src="https://cdn.tailwindcss.com" />
        <div className="flex h-screen bg-white">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-white border-r border-gray-200 transition-all duration-300`}>
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <img src="/logo.jpg" alt="zehanx AI" className="w-8 h-8 rounded" />
              {sidebarOpen && (
                <div>
                  <h2 className="font-semibold text-gray-900">zehanx AI</h2>
                  <p className="text-sm text-gray-500">Dashboard</p>
                </div>
              )}
            </div>
          </div>
          
          <div className="p-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-full flex items-center gap-3 px-3 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              {sidebarOpen && <span>Menu</span>}
            </button>
          </div>

          <div className="px-4 pb-4">
            <button
              onClick={signOut}
              className="w-full flex items-center gap-3 px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              {sidebarOpen && <span>Sign Out</span>}
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-auto">
          <div className="p-8">
            {/* Header */}
            <div className="mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome to zehanx AI</h1>
              <p className="text-gray-600">Choose an AI model to get started</p>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-sm font-medium text-gray-500 mb-2">Total Requests</h3>
                <p className="text-2xl font-bold text-gray-900">{userStats.totalRequests}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-sm font-medium text-gray-500 mb-2">Daily Requests</h3>
                <p className="text-2xl font-bold text-gray-900">{userStats.dailyRequests}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-sm font-medium text-gray-500 mb-2">Monthly Requests</h3>
                <p className="text-2xl font-bold text-gray-900">{userStats.monthlyRequests}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-sm font-medium text-gray-500 mb-2">Plan</h3>
                <p className="text-2xl font-bold text-gray-900">{userStats.isPremium ? "Pro" : "Free"}</p>
              </div>
            </div>

            {/* AI Models Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {aiModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all cursor-pointer hover:border-blue-300"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <span className="text-3xl">{model.icon}</span>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                      <span className="text-sm text-blue-600 bg-blue-100 px-2 py-1 rounded">
                        {model.category}
                      </span>
                    </div>
                  </div>
                  <p className="text-gray-600 text-sm mb-4">{model.description}</p>
                  <div className="text-xs text-gray-500">
                    Powered by zehanx AI
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      </>
    );
  }

  // Chat interface for selected model
  const currentModel = aiModels.find(m => m.id === selectedModel);

  return (
    <>
      <Script src="https://cdn.tailwindcss.com" />
      <div className="flex h-screen bg-white">
      {/* Chat Interface - Using the exact design you provided */}
      <div className="flex h-full w-full flex-col items-start">
        {/* Header */}
        <div className="w-full border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSelectedModel(null)}
                className="text-blue-600 hover:text-blue-800 text-sm"
              >
                ← Back to Dashboard
              </button>
              <span className="text-lg font-semibold text-gray-900">{currentModel?.name}</span>
            </div>
            <button
              onClick={signOut}
              className="text-red-600 hover:text-red-800 text-sm"
            >
              Sign Out
            </button>
          </div>
        </div>

        {messages.length === 0 ? (
          /* Welcome Screen */
          <div className="flex w-full grow shrink-0 basis-0 flex-col items-center justify-center gap-4 bg-white px-6 py-6">
            <div className="flex flex-col items-center justify-center gap-12">
              <img
                className="h-12 w-12 flex-none object-cover rounded-lg"
                src="/logo.jpg"
                alt="zehanx AI"
              />
              <div className="flex flex-wrap items-center justify-center gap-4">
                <div 
                  className="flex w-40 flex-none flex-col items-start gap-4 self-stretch rounded-md border border-solid border-gray-200 bg-white px-4 py-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => setInputMessage("Create an image for my slide deck")}
                >
                  <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="line-clamp-2 w-full text-sm text-gray-600">
                    Create an image for my slide deck
                  </span>
                </div>
                <div 
                  className="flex w-40 flex-none flex-col items-start gap-4 self-stretch rounded-md border border-solid border-gray-200 bg-white px-4 py-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => setInputMessage("Thank my interviewer")}
                >
                  <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  <span className="line-clamp-2 w-full text-sm text-gray-600">
                    Thank my interviewer
                  </span>
                </div>
                <div 
                  className="flex w-40 flex-none flex-col items-start gap-4 self-stretch rounded-md border border-solid border-gray-200 bg-white px-4 py-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => setInputMessage("Plan a relaxing day")}
                >
                  <svg className="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                  <span className="line-clamp-2 w-full text-sm text-gray-600">
                    Plan a relaxing day
                  </span>
                </div>
                <div 
                  className="flex w-40 flex-none flex-col items-start gap-4 self-stretch rounded-md border border-solid border-gray-200 bg-white px-4 py-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => setInputMessage("Explain nostalgia like I'm 5")}
                >
                  <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
                  </svg>
                  <span className="line-clamp-2 w-full text-sm text-gray-600">
                    Explain nostalgia like I'm 5
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Chat Messages */
          <div className="flex-1 overflow-y-auto p-6 w-full">
            <div className="max-w-4xl mx-auto space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.isUser
                        ? "bg-blue-500 text-white"
                        : "bg-gray-100 border border-gray-200 text-gray-900"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                    <p className="text-xs mt-1 opacity-70">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 border border-gray-200 text-gray-900 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                      </div>
                      <span className="text-sm text-gray-500">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="flex w-full flex-col items-center justify-center gap-3 px-4 py-4 border-t border-gray-200">
          <div className="flex w-full max-w-[768px] items-center justify-center gap-2 rounded-full bg-gray-100 px-2 py-2">
            <button className="p-2 rounded-full hover:bg-gray-200 transition-colors">
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.586-6.586a2 2 0 00-2.828-2.828z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4" />
              </svg>
            </button>
            <input
              className="h-auto grow shrink-0 basis-0 bg-transparent border-none outline-none px-4 py-2 text-gray-700 placeholder-gray-500"
              placeholder="Ask me anything"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button 
              disabled={!inputMessage.trim() || isLoading}
              onClick={() => handleSendMessage()}
              className="p-2 rounded-full bg-blue-500 text-white hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </button>
          </div>
          <span className="text-xs text-gray-500">
            zehanx AI can make mistakes. Check important info.
          </span>
        </div>
      </div>
    </div>
    </>
  );
}