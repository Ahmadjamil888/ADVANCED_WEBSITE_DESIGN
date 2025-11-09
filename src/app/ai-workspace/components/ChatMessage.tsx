'use client';

import { useState, useEffect } from 'react';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
  files?: string[];
}

export function ChatMessage({ role, content, isStreaming, files }: ChatMessageProps) {
  const [displayedContent, setDisplayedContent] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (isStreaming && currentIndex < content.length) {
      const timer = setTimeout(() => {
        setDisplayedContent(content.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 10);
      return () => clearTimeout(timer);
    } else {
      setDisplayedContent(content);
    }
  }, [content, currentIndex, isStreaming]);

  if (role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[80%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3">
          <p className="text-sm leading-relaxed">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 mb-4">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      <div className="flex-1 max-w-[80%]">
        <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
          <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
            {displayedContent}
            {isStreaming && currentIndex < content.length && (
              <span className="inline-block w-1 h-4 bg-blue-500 ml-1 animate-pulse" />
            )}
          </div>
          
          {files && files.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <div className="text-xs text-gray-400 mb-2">Generated Files:</div>
              <div className="flex flex-wrap gap-2">
                {files.map((file, idx) => (
                  <div
                    key={idx}
                    className="px-2 py-1 bg-gray-900 rounded text-xs font-mono text-green-400 border border-gray-700"
                  >
                    📄 {file}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
