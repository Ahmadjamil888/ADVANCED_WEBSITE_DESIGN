import { getSupabaseOrThrow } from "@/lib/supabase";
import type { Database } from "@/lib/database.types";

// Type definitions
export type Chat = Database["public"]["Tables"]["chats"]["Row"];
export type Message = Database["public"]["Tables"]["messages"]["Row"];
export type AIModel = Database["public"]["Tables"]["ai_models"]["Row"];
export type AIModelInsert = Database["public"]["Tables"]["ai_models"]["Insert"];
export type ChatInsert = Database["public"]["Tables"]["chats"]["Insert"];
export type MessageInsert = Database["public"]["Tables"]["messages"]["Insert"];

// Extended Message type with optional fields
export type ExtendedMessage = Message & {
  tokens_used?: number;
  eventId?: string;
};

// Chat management functions
export async function loadChats(userId: string): Promise<Chat[]> {
  try {
    const supabase = getSupabaseOrThrow();
    const { data: allChats, error } = await supabase.from('chats').select('*');
    if (error) throw error;
    if (allChats) {
      return (allChats as Chat[])
        .filter((chat: Chat) => chat.user_id === userId)
        .sort((a: Chat, b: Chat) => new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime());
    }
    return [];
  } catch (err) {
    console.error('Error loading chats:', err);
    return [];
  }
}

export async function loadMessages(chatId: string): Promise<Message[]> {
  try {
    const supabase = getSupabaseOrThrow();
    const { data: allMessages, error } = await supabase.from('messages').select('*');
    if (error) throw error;
    if (allMessages) {
      return allMessages
        .filter((msg: Message) => msg.chat_id === chatId)
        .sort((a: Message, b: Message) => new Date(a.created_at || 0).getTime() - new Date(b.created_at || 0).getTime());
    }
    return [];
  } catch (err) {
    console.error('Error loading messages:', err);
    return [];
  }
}

export async function createNewChat(userId: string): Promise<Chat | null> {
  try {
    const supabase = getSupabaseOrThrow();
    const newChatData: ChatInsert = {
      user_id: userId,
      title: 'New Chat',
      mode: 'models',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    const { data, error } = await (supabase.from('chats').insert as any)(newChatData).select().single();
    if (error) throw error;
    return data as Chat;
  } catch (error) {
    console.error('Error in createNewChat:', error);
    return null;
  }
}

export async function deleteChat(chatId: string): Promise<boolean> {
  try {
    const supabase = getSupabaseOrThrow();
    // Delete all messages for this chat first
    // Delete all messages for this chat first
    const { error: msgError } = await supabase.from('messages').delete().eq('chat_id', chatId);
    if (msgError) throw msgError;

    // Then delete the chat
    const { error } = await supabase.from('chats').delete().eq('id', chatId);
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Error in deleteChat:', error);
    return false;
  }
}

export async function updateChat(chatId: string, updates: Partial<Chat>): Promise<boolean> {
  try {
    const supabase = getSupabaseOrThrow();
    const updateData = {
      ...updates,
      updated_at: new Date().toISOString()
    };
    const { error } = await (supabase.from('chats').update as any)(updateData).eq('id', chatId);
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Error updating chat:', error);
    return false;
  }
}

// Message management functions
export async function insertMessage(messageData: MessageInsert): Promise<Message | null> {
  try {
    const supabase = getSupabaseOrThrow();
    const { data, error } = await (supabase.from('messages').insert as any)(messageData).select().single();
    if (error) throw error;
    return data as Message;
  } catch (error) {
    console.error('Error inserting message:', error);
    return null;
  }
}

export async function updateMessage(messageId: string, updates: Partial<Message>): Promise<boolean> {
  try {
    const supabase = getSupabaseOrThrow();
    const { error } = await (supabase.from('messages').update as any)(updates).eq('id', messageId);
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Error updating message:', error);
    return false;
  }
}

// AI Model management functions
export async function saveAIModel(modelData: AIModelInsert): Promise<AIModel | null> {
  try {
    const supabase = getSupabaseOrThrow();
    const { data, error } = await (supabase.from('ai_models').insert as any)(modelData).select().single();
    if (error) throw error;
    return data as AIModel;
  } catch (error) {
    console.error('Error saving AI model:', error);
    return null;
  }
}

// Helper functions for AI responses
export function generateNaturalResponse(prompt: string): string {
  const lowerPrompt = prompt.toLowerCase();
  
  if (lowerPrompt.includes('sentiment') || lowerPrompt.includes('emotion')) {
    return "Perfect! I understand you want to build a Sentiment Analysis model. Let me create that for you right now! I'll analyze your requirements, find the best model architecture, get some great training data, and build everything from scratch. This is going to be exciting! 🚀";
  } else if (lowerPrompt.includes('image') || lowerPrompt.includes('photo')) {
    return "Awesome! An Image Classification model - I love working with computer vision! Let me build you something amazing. I'll use a Vision Transformer, find perfect image datasets, and create a beautiful interface for testing. Let's make this happen! 🖼️✨";
  } else if (lowerPrompt.includes('chat') || lowerPrompt.includes('conversation')) {
    return "Fantastic! A Conversational AI model - this is going to be so cool! I'll build you a smart chatbot that can have natural conversations. I'll use the latest language models and create an interactive interface. Ready to bring your AI to life! 💬🤖";
  } else {
    return "Perfect! I understand you want to build a Text Classification model. Let me create that for you right now! I'll analyze your requirements, find the best model architecture, get some great training data, and build everything from scratch. This is going to be exciting! 🚀";
  }
}

export function generateAIThoughts(prompt: string): string {
  const lowerPrompt = prompt.toLowerCase();
  let thoughts = "Let me analyze this request...\n\n";
  
  if (lowerPrompt.includes('sentiment') || lowerPrompt.includes('emotion') || lowerPrompt.includes('feeling')) {
    thoughts += "🎯 I can see you want sentiment analysis!\n";
    thoughts += "💭 This is perfect for understanding emotions in text\n";
    thoughts += "🔍 I'll use a RoBERTa model - it's excellent for sentiment\n";
    thoughts += "📊 I'll find some great movie review data for training\n";
  } else if (lowerPrompt.includes('image') || lowerPrompt.includes('photo') || lowerPrompt.includes('picture')) {
    thoughts += "🖼️ Ah, image classification! Exciting!\n";
    thoughts += "💭 I'll use a Vision Transformer for this\n";
    thoughts += "🔍 Perfect for recognizing objects and scenes\n";
    thoughts += "📊 I'll get some diverse image datasets\n";
  } else if (lowerPrompt.includes('chat') || lowerPrompt.includes('conversation') || lowerPrompt.includes('bot')) {
    thoughts += "💬 A conversational AI! I love these!\n";
    thoughts += "💭 I'll use DialoGPT for natural conversations\n";
    thoughts += "🔍 This will be great for interactive responses\n";
    thoughts += "📊 I'll train on dialogue datasets\n";
  } else {
    thoughts += "📝 Looks like text classification!\n";
    thoughts += "💭 I'll use BERT - it's reliable and accurate\n";
    thoughts += "🔍 Perfect for categorizing and understanding text\n";
    thoughts += "📊 I'll find relevant training data\n";
  }
  
  thoughts += "\n🚀 This is going to be awesome! Let me get started...";
  return thoughts;
}

// Training and deployment functions
export async function startTrainingProcess(
  eventId: string,
  prompt: string,
  chatId: string,
  userId: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const response = await fetch('/api/ai-workspace/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chatId,
        prompt,
        mode: 'models',
        userId,
        eventId,
        useE2B: true
      })
    });

    const data = await response.json();
    
    if (data.error) {
      return { success: false, error: data.error };
    }

    return { success: true };
  } catch (error) {
    console.error('Training start error:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

export async function pollTrainingStatus(eventId: string): Promise<any> {
  try {
    const response = await fetch(`/api/ai-workspace/status/${eventId}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-cache'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Polling error:', error);
    throw error;
  }
}

// URL validation helpers
export function isE2bUrl(url: unknown): url is string {
  return typeof url === 'string' && /\.e2b\.dev(\/?|$)/.test(url);
}

export function isFallbackLocalUrl(url: unknown): url is string {
  return typeof url === 'string' && url.startsWith('/e2b-fallback/');
}

// Create a complete message object with all required fields
export function createMessage(
  chatId: string,
  role: 'user' | 'assistant' | 'system',
  content: string,
  options?: {
    model_used?: string | null;
    metadata?: any;
    tokens_used?: number;
  }
): MessageInsert {
  return {
    chat_id: chatId,
    role,
    content,
    model_used: options?.model_used || null,
    metadata: options?.metadata || {},
    created_at: new Date().toISOString()
  };
}

// Create a complete chat object
export function createChatObject(
  userId: string,
  title: string,
  mode: string = 'models'
): ChatInsert {
  return {
    user_id: userId,
    title,
    mode,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };
}
