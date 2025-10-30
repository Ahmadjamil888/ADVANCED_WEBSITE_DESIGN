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

const proFeatures = [
  {
    id: "crypto-bot",
    name: "Crypto Bot",
    description: "AI-powered cryptocurrency trading insights",
    icon: "₿",
    category: "Finance",
    isPro: true
  },
  {
    id: "reconnaissance",
    name: "Reconnaissance",
    description: "Advanced data analysis and reconnaissance",
    icon: "🕵️",
    category: "Security",
    isPro: true
  },
  {
    id: "image-gen",
    name: "Image Generator",
    description: "Generate images from text descriptions",
    icon: "🎨",
    category: "Creative",
    isPro: true
  },
  {
    id: "video-gen",
    name: "Video Generator",
    description: "Create videos from text and images",
    icon: "🎬",
    category: "Creative",
    isPro: true
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
    if (!user) return;

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
      // Call Gemini API
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

      // Update user stats
      await supabase
        .from("users")
        .update({
          total_requests: userStats.totalRequests + 1,
          daily_requests: userStats.dailyRequests + 1,
          monthly_requests: userStats.monthlyRequests + 1,
          last_activity: new Date().toISOString()
        })
        .eq("id", user?.id);

      // Log usage
      await supabase.from("model_usage").insert({
        user_id: user?.id,
        model_name: currentModel,
        prompt_tokens: messageText.length,
        completion_tokens: data.response.length,
        total_tokens: messageText.length + data.response.length,
        cost_credits: 1
      });

      fetchUserStats();
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
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  if (!selectedModel) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">zehanx AI Dashboard</h1>
              <p className="text-gray-600 mt-2">Welcome back, {user.email}</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-600">
                <div>Daily: {userStats.dailyRequests}/100</div>
                <div>Monthly: {userStats.monthlyRequests}/1000</div>
              </div>
              <button
                onClick={signOut}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                Sign Out
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-sm font-medium text-gray-500">Total Requests</h3>
              <p className="text-2xl font-bold text-gray-900">{userStats.totalRequests}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-sm font-medium text-gray-500">Daily Requests</h3>
              <p className="text-2xl font-bold text-gray-900">{userStats.dailyRequests}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-sm font-medium text-gray-500">Monthly Requests</h3>
              <p className="text-2xl font-bold text-gray-900">{userStats.monthlyRequests}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-sm font-medium text-gray-500">Plan</h3>
              <p className="text-2xl font-bold text-gray-900">{userStats.isPremium ? "Pro" : "Free"}</p>
            </div>
          </div>

          {/* AI Models */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Available AI Models</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {aiModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer border border-gray-200 hover:border-blue-300"
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
                  <p className="text-gray-600 text-sm">{model.description}</p>
                  <div className="mt-4 text-xs text-gray-500">
                    Powered by zehanx AI
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pro Features */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Pro Features (Coming Soon)</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {proFeatures.map((feature) => (
                <div
                  key={feature.id}
                  className="bg-gradient-to-br from-purple-50 to-blue-50 p-6 rounded-lg shadow-sm border border-purple-200 relative"
                >
                  <div className="absolute top-2 right-2">
                    <span className="text-xs bg-purple-500 text-white px-2 py-1 rounded-full">PRO</span>
                  </div>
                  <div className="flex items-center gap-4 mb-4">
                    <span className="text-3xl">{feature.icon}</span>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{feature.name}</h3>
                      <span className="text-sm text-purple-600 bg-purple-100 px-2 py-1 rounded">
                        {feature.category}
                      </span>
                    </div>
                  </div>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                  <div className="mt-4 text-xs text-gray-500">
                    Coming Soon
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* API Section */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <h2 className="text-xl font-bold text-yellow-800 mb-2">API Access</h2>
            <p className="text-yellow-700">
              API functionality is coming soon! You'll be able to integrate zehanx AI models into your applications.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const currentModel = aiModels.find(m => m.id === selectedModel);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-white shadow-sm border-r border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={() => setSelectedModel(null)}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            ← Back to Dashboard
          </button>
          <h2 className="text-lg font-semibold text-gray-900 mt-2">{currentModel?.name}</h2>
          <p className="text-sm text-gray-600">{currentModel?.description}</p>
        </div>
        <div className="p-4">
          <div className="text-xs text-gray-500 mb-2">Usage Today</div>
          <div className="text-sm font-medium text-gray-900">{userStats.dailyRequests}/100 requests</div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <span className="text-4xl mb-4 block">{currentModel?.icon}</span>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{currentModel?.name}</h3>
                <p className="text-gray-600">Start a conversation with this AI model</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4 max-w-4xl mx-auto">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.isUser
                        ? "bg-blue-500 text-white"
                        : "bg-white border border-gray-200 text-gray-900"
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
                  <div className="bg-white border border-gray-200 text-gray-900 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
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
          )}
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center space-x-4">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                placeholder={`Ask ${currentModel?.name}...`}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                onClick={() => handleSendMessage()}
                disabled={!inputMessage.trim() || isLoading}
                className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Powered by zehanx AI • {currentModel?.name}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}