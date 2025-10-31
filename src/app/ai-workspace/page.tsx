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
  Spinner,
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
  onDownloadFiles,
  hasHfToken 
}: { 
  message: Message;
  onHuggingFaceUpload: (eventId: string, hfToken?: string) => void;
  onDownloadFiles: (eventId: string) => void;
  hasHfToken: boolean;
}) => {
  const [isUploading, setIsUploading] = useState(false);

  const isModelGenerated = message.role === 'assistant' && 
    (message.content.includes('Generation Complete') || message.content.includes('AI Model Successfully Generated')) && 
    message.eventId;

  return (
    <Flex horizontal={message.role === 'user' ? 'end' : 'start'}>
      <Card
        padding="m"
        background={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
        style={{ maxWidth: '85%' }}
      >
        <Flex direction="column" gap="m">
          <Text 
            variant="body-default-m" 
            onBackground={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
            style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
          >
            {message.content}
          </Text>
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
  const [currentMode, setCurrentMode] = useState<string>('models');
  const [isLoading, setIsLoading] = useState(false);
  const [pendingModels, setPendingModels] = useState<Set<string>>(new Set());
  const [userHfToken, setUserHfToken] = useState<string>('');

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
      return;
    }

    if (user) {
      loadChats();
      loadUserHfToken();
    }
  }, [user, loading, router]);

  const loadUserHfToken = async () => {
    if (!user) return;
    
    try {
      const response = await fetch(`/api/user/hf-token?userId=${user.id}`);
      const data = await response.json();
      if (data.hasToken) {
        setUserHfToken(data.token);
      }
    } catch (error) {
      console.error('Error loading HF token:', error);
    }
  };

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
        title: 'New AI Model',
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
      await supabase.from('messages').delete().eq('chat_id', chatId);
      await supabase.from('chats').delete().eq('id', chatId);
      
      setChats(prev => prev.filter(chat => chat.id !== chatId));
      
      if (currentChat?.id === chatId) {
        setCurrentChat(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const handleHuggingFaceUpload = async (eventId: string, providedToken?: string) => {
    try {
      let tokenToUse = providedToken || userHfToken;
      
      if (!tokenToUse) {
        const token = prompt('Please enter your Hugging Face token:');
        if (!token) return;
        tokenToUse = token;
        
        await fetch('/api/user/hf-token', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            userId: user?.id,
            hfToken: token
          })
        });
        setUserHfToken(token);
      }

      const response = await fetch('/api/ai-workspace/deploy-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          eventId,
          hfToken: tokenToUse,
          userId: user?.id
        })
      });

      const data = await response.json();
      
      if (data.success) {
        const successMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: `# 🎉 Model Deployed Successfully!\n\n🔗 **Your Model:** [${data.repoUrl}](${data.repoUrl})\n\nYour AI model is now live on Hugging Face Hub!`,
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

        const completionMessage: Message = {
          id: `completion-${eventId}`,
          role: 'assistant',
          content: `# 🎉 **${data.model.name} - Generation Complete!**\n\nYour AI model has been successfully generated and is ready for use!\n\n## 📊 **Model Details**\n- **Name:** ${data.model.name}\n- **Type:** ${data.model.type.replace('-', ' ').toUpperCase()}\n- **Framework:** ${data.model.framework.toUpperCase()}\n- **Dataset:** ${data.model.dataset}\n- **Status:** ✅ Ready for Training & Deployment\n\n🚀 Your model is production-ready and follows industry best practices!`,
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

  const sendMessage = async (content: string) => {
    if (!currentChat || !supabase || !user || isLoading) return;

    setIsLoading(true);

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      await supabase.from('messages').insert({
        chat_id: currentChat.id,
        role: 'user',
        content
      });

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
        chat_id: currentChat.id,
        role: 'assistant',
        content: data.response,
        tokens_used: data.tokens_used,
        model_used: data.model_used
      });

      await supabase
        .from('chats')
        .update({ updated_at: new Date().toISOString() })
        .eq('id', currentChat.id);

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
        width={18}
        background="neutral-strong" 
        padding="s"
        gap="s"
        style={{ height: '100vh', overflow: 'hidden', minWidth: '280px' }}
      >
        <Flex direction="column" gap="xs" padding="s">
          <Heading variant="heading-strong-s" onBackground="neutral-strong">
            AI Workspace
          </Heading>
          <Text variant="body-default-xs" onBackground="neutral-medium">
            Build AI models with ease
          </Text>
          
          <Button
            onClick={createNewChat}
            variant="secondary"
            size="s"
            fillWidth
          >
            + New Model
          </Button>
        </Flex>
        
        <Flex flex={1} direction="column" gap="xs" style={{ overflow: 'auto', minHeight: 0 }} padding="s">
          <Text variant="body-default-xs" onBackground="neutral-medium">
            Recent Projects
          </Text>
          {chats.length === 0 ? (
            <Text variant="body-default-xs" onBackground="neutral-medium" style={{ fontStyle: 'italic' }}>
              No projects yet
            </Text>
          ) : (
            chats.map((chat) => (
              <Card
                key={chat.id}
                padding="s"
                background={currentChat?.id === chat.id ? "neutral-medium" : "neutral-weak"}
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
                    style={{ marginLeft: '8px', flexShrink: 0, minWidth: '24px', height: '24px' }}
                  >
                    ×
                  </Button>
                </Flex>
              </Card>
            ))
          )}
        </Flex>
        
        <Flex direction="column" gap="xs" padding="s" borderTop="neutral-medium">
          {user && (
            <Flex vertical="center" gap="s">
              <Avatar size="s" src={user.user_metadata?.avatar_url} />
              <Text variant="body-default-xs" onBackground="neutral-medium" style={{ flex: 1 }}>
                {user.email}
              </Text>
            </Flex>
          )}
          <Button
            onClick={signOut}
            variant="tertiary"
            size="s"
            fillWidth
          >
            🚪 Sign out
          </Button>
        </Flex>
      </Flex>

      {/* Main Content Area */}
      <Flex flex={1} direction="column" background="neutral-weak" style={{ height: '100vh', overflow: 'hidden' }}>
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
                  hasHfToken={!!userHfToken}
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
              
              {pendingModels.size > 0 && (
                <Flex horizontal="start">
                  <Card padding="m" background="brand-weak" border="brand-medium">
                    <Flex center gap="s">
                      <Spinner size="s" />
                      <Text variant="body-default-s" onBackground="brand-weak">
                        🔧 Generating AI model... This may take 1-2 minutes. Please keep this chat open.
                      </Text>
                    </Flex>
                  </Card>
                </Flex>
              )}
            </Flex>
          )}
          
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