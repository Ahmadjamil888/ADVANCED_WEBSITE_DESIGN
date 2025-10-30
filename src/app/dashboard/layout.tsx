export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <style>{`
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: white;
            color: #111827;
            height: 100vh;
            overflow: hidden;
          }
          
          .dashboard-container {
            display: flex;
            height: 100vh;
            background: white;
          }
          
          .sidebar {
            background: white;
            border-right: 1px solid #e5e7eb;
            transition: width 0.3s ease;
          }
          
          .sidebar.open {
            width: 16rem;
          }
          
          .sidebar.closed {
            width: 4rem;
          }
          
          .sidebar-header {
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
          }
          
          .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
          }
          
          .sidebar-logo {
            width: 2rem;
            height: 2rem;
            border-radius: 0.25rem;
          }
          
          .sidebar-title {
            font-weight: 600;
            color: #111827;
          }
          
          .sidebar-subtitle {
            font-size: 0.875rem;
            color: #6b7280;
          }
          
          .sidebar-menu {
            padding: 1rem;
          }
          
          .menu-button {
            width: 100%;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            color: #374151;
            background: none;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s;
          }
          
          .menu-button:hover {
            background-color: #f3f4f6;
          }
          
          .menu-icon {
            width: 1.25rem;
            height: 1.25rem;
          }
          
          .sign-out-button {
            width: 100%;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            color: #dc2626;
            background: none;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: auto;
          }
          
          .sign-out-button:hover {
            background-color: #fef2f2;
          }
          
          .main-content {
            flex: 1;
            overflow: auto;
          }
          
          .loading-container {
            display: flex;
            height: 100vh;
            width: 100%;
            align-items: center;
            justify-content: center;
            background: white;
          }
          
          .loading-text {
            font-size: 1.125rem;
            color: #111827;
          }
          
          .dashboard-content {
            padding: 2rem;
          }
          
          .dashboard-header {
            margin-bottom: 2rem;
          }
          
          .dashboard-title {
            font-size: 1.875rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.5rem;
          }
          
          .dashboard-subtitle {
            color: #6b7280;
          }
          
          .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
          }
          
          .stat-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
          }
          
          .stat-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: #6b7280;
            margin-bottom: 0.5rem;
          }
          
          .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #111827;
          }
          
          .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
          }
          
          .model-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.2s;
          }
          
          .model-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-color: #93c5fd;
          }
          
          .model-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
          }
          
          .model-icon {
            font-size: 1.875rem;
          }
          
          .model-info h3 {
            font-size: 1.125rem;
            font-weight: 600;
            color: #111827;
          }
          
          .model-category {
            font-size: 0.875rem;
            color: #2563eb;
            background-color: #dbeafe;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
          }
          
          .model-description {
            color: #6b7280;
            font-size: 0.875rem;
            margin-bottom: 1rem;
          }
          
          .model-footer {
            font-size: 0.75rem;
            color: #6b7280;
          }
          
          .chat-container {
            display: flex;
            height: 100vh;
            width: 100%;
            flex-direction: column;
            align-items: flex-start;
            position: relative;
          }
          
          .chat-header {
            width: 100%;
            border-bottom: 1px solid #e5e7eb;
            padding: 1rem;
          }
          
          .chat-header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
          }
          
          .chat-nav {
            display: flex;
            align-items: center;
            gap: 0.75rem;
          }
          
          .back-button {
            color: #2563eb;
            background: none;
            border: none;
            font-size: 0.875rem;
            cursor: pointer;
          }
          
          .back-button:hover {
            color: #1d4ed8;
          }
          
          .chat-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: #111827;
          }
          
          .welcome-screen {
            display: flex;
            width: 100%;
            flex: 1;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            background: white;
            padding: 1.5rem;
          }
          
          .welcome-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 3rem;
          }
          
          .welcome-logo {
            height: 3rem;
            width: 3rem;
            border-radius: 0.5rem;
            object-fit: cover;
          }
          
          .suggestions-grid {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 1rem;
          }
          
          .suggestion-card {
            display: flex;
            width: 10rem;
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
            border-radius: 0.375rem;
            border: 1px solid #e5e7eb;
            background: white;
            padding: 1rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            cursor: pointer;
            transition: box-shadow 0.2s;
          }
          
          .suggestion-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          }
          
          .suggestion-icon {
            width: 1.5rem;
            height: 1.5rem;
          }
          
          .suggestion-text {
            width: 100%;
            font-size: 0.875rem;
            color: #6b7280;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }
          
          .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            width: 100%;
            padding-bottom: 120px;
          }
          
          .messages-container {
            max-width: 64rem;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
          }
          
          .message {
            display: flex;
          }
          
          .message.user {
            justify-content: flex-end;
          }
          
          .message.ai {
            justify-content: flex-start;
          }
          
          .message-bubble {
            max-width: 20rem;
            padding: 1rem;
            border-radius: 0.5rem;
          }
          
          .message-bubble.user {
            background-color: #3b82f6;
            color: white;
          }
          
          .message-bubble.ai {
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            color: #111827;
          }
          
          .message-text {
            font-size: 0.875rem;
            white-space: pre-wrap;
          }
          
          .message-time {
            font-size: 0.75rem;
            margin-top: 0.25rem;
            opacity: 0.7;
          }
          
          .typing-indicator {
            display: flex;
            justify-content: flex-start;
          }
          
          .typing-bubble {
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            color: #111827;
            max-width: 20rem;
            padding: 1rem;
            border-radius: 0.5rem;
          }
          
          .typing-dots {
            display: flex;
            align-items: center;
            gap: 0.5rem;
          }
          
          .typing-dot {
            width: 0.5rem;
            height: 0.5rem;
            background-color: #9ca3af;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
          }
          
          .typing-dot:nth-child(1) { animation-delay: -0.32s; }
          .typing-dot:nth-child(2) { animation-delay: -0.16s; }
          
          @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
          }
          
          .typing-text {
            font-size: 0.875rem;
            color: #6b7280;
            margin-left: 0.5rem;
          }
          
          .chat-input-area {
            display: flex;
            width: 100%;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
            background: white;
            position: sticky;
            bottom: 0;
            z-index: 10;
          }
          
          .chat-input-container {
            display: flex;
            width: 100%;
            max-width: 48rem;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            border-radius: 25px;
            background-color: #f3f4f6;
            padding: 0.5rem;
            border: 1px solid #e5e7eb;
          }
          
          .input-button {
            padding: 0.5rem;
            border-radius: 50%;
            background: none;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 32px;
            min-height: 32px;
          }
          
          .input-button:hover {
            background-color: #e5e7eb;
          }
          
          .input-button svg {
            width: 1.25rem;
            height: 1.25rem;
            color: #6b7280;
          }
          
          .chat-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            padding: 0.75rem 1rem;
            color: #374151;
            font-size: 16px;
            min-height: 20px;
          }
          
          .chat-input::placeholder {
            color: #6b7280;
          }
          
          .send-button {
            padding: 0.75rem;
            border-radius: 50%;
            background-color: #3b82f6;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 40px;
            min-height: 40px;
          }
          
          .send-button:hover {
            background-color: #2563eb;
          }
          
          .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }
          
          .send-button svg {
            width: 1.25rem;
            height: 1.25rem;
          }
          
          .chat-disclaimer {
            font-size: 0.75rem;
            color: #6b7280;
          }
          
          @media (min-width: 1024px) {
            .messages-container {
              max-width: 48rem;
            }
            
            .message-bubble {
              max-width: 24rem;
            }
          }
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  )
}