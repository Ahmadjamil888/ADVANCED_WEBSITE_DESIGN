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
    <Flex fillWidth fillHeight direction="row" background="neutral-weak">
      {/* Sidebar */}
      <Flex 
        direction="column" 
        width={20}
        background="neutral-strong" 
        padding="l"
        gap="l"
      >
        <Flex direction="column" gap="s">
          <Heading variant="heading-strong-l" onBackground="neutral-strong">
            AI Workspace
          </Heading>
          <Text variant="body-default-s" onBackground="neutral-medium">
            Generate, train, and deploy AI models
          </Text>
          
          <Button
            onClick={createNewChat}
            variant="secondary"
            size="m"
            fillWidth
          >
            + New chat
          </Button>
        </Flex>
        
        <Flex flex={1} direction="column" gap="s">
          <Text variant="body-default-s" onBackground="neutral-medium">
            Recent chats will appear here
          </Text>
          {chats.map((chat) => (
            <Card
              key={chat.id}
              padding="s"
              background="neutral-medium"
              style={{ cursor: 'pointer' }}
              onClick={() => {
                setCurrentChat(chat);
                loadMessages(chat.id);
              }}
            >
              <Text variant="body-default-s" onBackground="neutral-medium">
                {chat.title}
              </Text>
            </Card>
          ))}
        </Flex>
        
        <Button
          onClick={signOut}
          variant="tertiary"
          size="m"
          fillWidth
        >
          Sign out
        </Button>
      </Flex>

      {/* Main Content Area */}
      <Flex flex={1} direction="column" background="neutral-weak">
        {/* Header */}
        <Flex 
          padding="l" 
          borderBottom="neutral-medium" 
          direction="column" 
          gap="xs"
        >
          <Heading variant="heading-strong-m" onBackground="neutral-weak">
            AI Model Generator
          </Heading>
          <Text variant="body-default-s" onBackground="neutral-medium">
            Describe the AI model you want to create
          </Text>
        </Flex>
        
        {/* Chat Area */}
        <Flex flex={1} direction="column">
          {messages.length === 0 ? (
            <Flex flex={1} center padding="xl">
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
            <Flex flex={1} direction="column" padding="l" gap="l" style={{ overflowY: 'auto' }}>
              {messages.map((message) => (
                <Flex 
                  key={message.id} 
                  horizontal={message.role === 'user' ? 'end' : 'start'}
                >
                  <Card
                    padding="m"
                    background={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
                    style={{ maxWidth: '70%' }}
                  >
                    <Text 
                      variant="body-default-m" 
                      onBackground={message.role === 'user' ? 'brand-medium' : 'neutral-medium'}
                      style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                    >
                      {message.content}
                    </Text>
                  </Card>
                </Flex>
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
          <Flex padding="l" borderTop="neutral-medium">
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