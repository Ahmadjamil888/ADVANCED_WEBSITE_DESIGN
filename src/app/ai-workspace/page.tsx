"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { 
  Flex, 
  Button, 
  Text, 
  Heading, 
  Input, 
  Card, 
  Avatar, 
  Icon, 
  Spinner,
  Badge,
  Textarea
} from "@/once-ui/components";

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

// MessageComponent for handling AI responses with actions
const MessageComponent = ({ 
  message, 
  onHuggingFaceUpload, 
  onDownloadFiles 
}: { 
  message: Message;
  onHuggingFaceUpload: (eventId: string, hfToken: string) => void;
  onDownloadFiles: (eventId: string) => void;
}) => {
  const [showHfDialog, setShowHfDialog] = useState(false);
  const [hfToken, setHfToken] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const isModelGenerated = message.role === 'assistant' && 
    message.content.includes('AI Model Successfully Generated') && 
    message.eventId;

  return (
    <Flex horizontal={message.role === 'user' ? 'end' : 'start'}>
      <Card
        padding="m"
        background={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
        style={{ maxWidth: '80%' }}
      >
        <Flex direction="column" gap="m">
          <Text 
            variant="body-default-m" 
            onBackground={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
            style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
          >
            {message.content}
          </Text>
          
          {isModelGenerated && (
            <Flex direction="column" gap="s">
              <Text variant="body-strong-s" onBackground="neutral-medium">
                🚀 Your AI model is ready! What would you like to do?
              </Text>
              
              <Flex gap="s" wrap>
                <Button
                  variant="primary"
                  size="s"
                  onClick={() => setShowHfDialog(true)}
                  disabled={isUploading}
                >
                  📤 Deploy to Hugging Face
                </Button>
                
                <Button
                  variant="secondary"
                  size="s"
                  onClick={() => message.eventId && onDownloadFiles(message.eventId)}
                >
                  💾 Download Files
                </Button>
              </Flex>

              {showHfDialog && (
                <Card padding="m" background="neutral-weak" border="neutral-medium">
                  <Flex direction="column" gap="s">
                    <Text variant="body-strong-s" onBackground="neutral-weak">
                      Deploy to Hugging Face Hub
                    </Text>
                    <Text variant="body-default-xs" onBackground="neutral-medium">
                      Enter your Hugging Face token to deploy your model to the Hub
                    </Text>
                    
                    <Input
                      id="hf-token"
                      label="Hugging Face Token"
                      placeholder="hf_..."
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                    />
                    
                    <Flex gap="s">
                      <Button
                        variant="primary"
                        size="s"
                        onClick={async () => {
                          if (hfToken.trim() && message.eventId) {
                            setIsUploading(true);
                            await onHuggingFaceUpload(message.eventId, hfToken.trim());
                            setShowHfDialog(false);
                            setHfToken('');
                            setIsUploading(false);
                          }
                        }}
                        disabled={!hfToken.trim() || isUploading}
                      >
                        {isUploading ? 'Deploying...' : 'Deploy'}
                      </Button>
                      
                      <Button
                        variant="tertiary"
                        size="s"
                        onClick={() => {
                          setShowHfDialog(false);
                          setHfToken('');
                        }}
                      >
                        Cancel
                      </Button>
                    </Flex>
                  </Flex>
                </Card>
              )}
            </Flex>
          )}
        </Flex>
      </Card>
    </Flex>
  );
};

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

  const deleteChat = async (chatId: string) => {
    if (!supabase) return;

    try {
      // Delete messages first
      await supabase.from('messages').delete().eq('chat_id', chatId);
      
      // Delete chat
      await supabase.from('chats').delete().eq('id', chatId);
      
      // Update UI
      setChats(prev => prev.filter(chat => chat.id !== chatId));
      
      // If this was the current chat, clear it
      if (currentChat?.id === chatId) {
        setCurrentChat(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const handleHuggingFaceUpload = async (eventId: string, hfToken: string) => {
    try {
      const response = await fetch('/api/ai-workspace/deploy-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          eventId,
          hfToken,
          userId: user?.id
        })
      });

      const data = await response.json();
      
      if (data.success) {
        // Add success message to chat
        const successMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: `🎉 **Model Successfully Deployed to Hugging Face!**\n\n🔗 **Your Model:** ${data.repoUrl}\n\nYour AI model is now live and ready to use! You can share it with the community or integrate it into your applications.`,
          created_at: new Date().toISOString()
        };
        setMessages(prev => [...prev, successMessage]);
      }
    } catch (error) {
      console.error('Error uploading to Hugging Face:', error);
    }
  };

  const handleDownloadFiles = async (eventId: string) => {
    try {
      const response = await fetch(`/api/ai-workspace/download/${eventId}`);
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-model-${eventId}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading files:', error);
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

      // Add AI response to UI with eventId for actions
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        created_at: new Date().toISOString(),
        tokens_used: data.tokens_used,
        eventId: data.eventId // Store eventId for later actions
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
      <Flex 
        fillWidth 
        fillHeight 
        center
        background="neutral-weak"
      >
        <Flex direction="column" center gap="m">
          <Spinner size="l" />
          <Text variant="body-default-s" onBackground="neutral-weak">
            Loading AI Workspace...
          </Text>
        </Flex>
      </Flex>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <Flex fillWidth fillHeight direction="row" background="neutral-weak" style={{ height: '100vh', overflow: 'hidden' }}>
      {/* Sidebar */}
      <Flex 
        direction="column" 
        width={20}
        background="neutral-strong" 
        padding="m"
        gap="m"
        style={{ height: '100vh', overflow: 'hidden' }}
      >
        <Flex direction="column" gap="s">
          <Heading variant="heading-strong-m" onBackground="neutral-strong">
            AI Workspace
          </Heading>
          <Text variant="body-default-xs" onBackground="neutral-medium">
            Generate, train, and deploy AI models
          </Text>
          
          <Button
            onClick={createNewChat}
            variant="secondary"
            size="s"
            fillWidth
          >
            + New chat
          </Button>
        </Flex>
        
        <Flex flex={1} direction="column" gap="xs" style={{ overflow: 'auto', minHeight: 0 }}>
          <Text variant="body-default-xs" onBackground="neutral-medium">
            Recent chats
          </Text>
          {chats.map((chat) => (
            <Card
              key={chat.id}
              padding="xs"
              background="neutral-medium"
              style={{ cursor: 'pointer' }}
              onClick={() => {
                setCurrentChat(chat);
                loadMessages(chat.id);
              }}
            >
              <Flex horizontal="space-between" vertical="center">
                <Text variant="body-default-xs" onBackground="neutral-medium" style={{ 
                  overflow: 'hidden', 
                  textOverflow: 'ellipsis', 
                  whiteSpace: 'nowrap',
                  flex: 1
                }}>
                  {chat.title}
                </Text>
                <Button
                  variant="tertiary"
                  size="s"
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    deleteChat(chat.id);
                  }}
                  style={{ marginLeft: '8px', flexShrink: 0 }}
                >
                  ×
                </Button>
              </Flex>
            </Card>
          ))}
        </Flex>
        
        <Button
          onClick={signOut}
          variant="tertiary"
          size="s"
          fillWidth
        >
          Sign out
        </Button>
      </Flex>

      {/* Main Content Area */}
      <Flex flex={1} direction="column" background="neutral-weak" style={{ height: '100vh', overflow: 'hidden' }}>
        {/* Header */}
        <Flex 
          padding="m" 
          borderBottom="neutral-medium" 
          direction="column" 
          gap="xs"
          style={{ flexShrink: 0 }}
        >
          <Heading variant="heading-strong-m" onBackground="neutral-weak">
            AI Model Generator
          </Heading>
          <Text variant="body-default-s" onBackground="neutral-medium">
            Describe the AI model you want to create
          </Text>
        </Flex>
        
        {/* Chat Area */}
        <Flex flex={1} direction="column" style={{ minHeight: 0 }}>
          {messages.length === 0 ? (
            <Flex flex={1} center padding="l" style={{ overflow: 'auto' }}>
              <Flex direction="column" center gap="l" maxWidth={25}>
                <Avatar 
                  size="xl" 
                  src="/logo.jpg"
                />
                <Flex direction="column" center gap="s">
                  <Heading variant="heading-strong-l" onBackground="neutral-weak">
                    Ready to create amazing AI models?
                  </Heading>
                  <Text variant="body-default-m" onBackground="neutral-medium" align="center">
                    I can help you generate, train, and deploy custom AI models. Just describe what you want to build!
                  </Text>
                </Flex>
                
                <Flex direction="column" gap="s" fillWidth>
                  <Card padding="m" background="neutral-weak" style={{ cursor: 'pointer' }}>
                    <Flex direction="column" gap="xs">
                      <Text variant="body-strong-m" onBackground="neutral-weak">
                        🎯 Text Classification
                      </Text>
                      <Text variant="body-default-s" onBackground="neutral-medium">
                        Create a sentiment analysis model
                      </Text>
                    </Flex>
                  </Card>
                  <Card padding="m" background="neutral-weak" style={{ cursor: 'pointer' }}>
                    <Flex direction="column" gap="xs">
                      <Text variant="body-strong-m" onBackground="neutral-weak">
                        🖼️ Image Classification
                      </Text>
                      <Text variant="body-default-s" onBackground="neutral-medium">
                        Detect and classify objects in images
                      </Text>
                    </Flex>
                  </Card>
                  <Card padding="m" background="neutral-weak" style={{ cursor: 'pointer' }}>
                    <Flex direction="column" gap="xs">
                      <Text variant="body-strong-m" onBackground="neutral-weak">
                        🤖 Chatbot Model
                      </Text>
                      <Text variant="body-default-s" onBackground="neutral-medium">
                        Build a conversational AI assistant
                      </Text>
                    </Flex>
                  </Card>
                </Flex>
              </Flex>
            </Flex>
          ) : (
            <Flex flex={1} direction="column" padding="m" gap="m" style={{ overflow: 'auto', minHeight: 0 }}>
              {messages.map((message) => (
                <MessageComponent 
                  key={message.id} 
                  message={message}
                  onHuggingFaceUpload={handleHuggingFaceUpload}
                  onDownloadFiles={handleDownloadFiles}
                />
              ))}
              
              {isLoading && (
                <Flex horizontal="start">
                  <Card padding="m" background="neutral-medium">
                    <Flex center gap="s">
                      <Spinner size="s" />
                      <Text variant="body-default-s" onBackground="neutral-medium">
                        AI is thinking...
                      </Text>
                    </Flex>
                  </Card>
                </Flex>
              )}
            </Flex>
          )}
          
          {/* Input Area */}
          <Flex padding="m" borderTop="neutral-medium" style={{ flexShrink: 0 }}>
            <Flex direction="column" gap="s" fillWidth maxWidth={50} style={{ margin: '0 auto' }}>
              <Card padding="m" background="neutral-weak" border="neutral-medium">
                <Flex direction="column" gap="s">
                  <Textarea
                    id="ai-prompt"
                    label="AI Model Description"
                    labelAsPlaceholder
                    placeholder="Describe the AI model you want to create (e.g., 'Create a sentiment analysis model using BERT...')"
                    lines={3}
                    disabled={isLoading}
                    resize="none"
                  />
                  <Flex horizontal="end">
                    <Button
                      onClick={() => {
                        const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
                        if (textarea?.value.trim()) {
                          sendMessage(textarea.value.trim());
                          textarea.value = '';
                        }
                      }}
                      disabled={isLoading}
                      variant="primary"
                      size="m"
                    >
                      Send →
                    </Button>
                  </Flex>
                </Flex>
              </Card>
              <Text variant="body-default-xs" onBackground="neutral-medium" align="center">
                zehanx AI can generate, train, and deploy custom AI models. Always verify generated code before training.
              </Text>
            </Flex>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
}