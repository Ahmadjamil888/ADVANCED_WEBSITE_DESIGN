"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";

function AiChatHome() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
    }
  }, [user, loading, router]);

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

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  const handleSubmit = () => {
    if (inputValue.trim()) {
      // Handle message submission
      router.push(`/dashboard?model=assistant&prompt=${encodeURIComponent(inputValue)}`);
    }
  };

  return (
    <div className="flex h-screen w-full flex-col items-start bg-white">
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
              onClick={() => handleSuggestionClick("Create an image for my slide deck")}
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
              onClick={() => handleSuggestionClick("Thank my interviewer")}
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
              onClick={() => handleSuggestionClick("Plan a relaxing day")}
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
              onClick={() => handleSuggestionClick("Explain nostalgia like I'm 5")}
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
      <div className="flex w-full flex-col items-center justify-center gap-3 px-4 py-4">
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
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
          />
          <button 
            disabled={!inputValue.trim()}
            onClick={handleSubmit}
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
  );
}

export default AiChatHome;