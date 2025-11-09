// AI Client for Groq, Gemini, and DeepSeek
import { AIProvider } from './models';

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface StreamChunk {
  content: string;
  done: boolean;
}

export class AIClient {
  private provider: AIProvider;
  private apiKey: string;
  private modelId: string;

  constructor(provider: AIProvider, modelId: string) {
    this.provider = provider;
    this.modelId = modelId;
    
    // Get API key from environment
    switch (provider) {
      case 'groq':
        this.apiKey = process.env.GROQ_API_KEY || '';
        break;
      case 'gemini':
        this.apiKey = process.env.GEMINI_API_KEY || '';
        break;
      case 'deepseek':
        this.apiKey = process.env.DEEPSEEK_API_KEY || '';
        break;
    }
  }

  async *streamCompletion(messages: Message[]): AsyncGenerator<StreamChunk> {
    switch (this.provider) {
      case 'groq':
        yield* this.streamGroq(messages);
        break;
      case 'gemini':
        yield* this.streamGemini(messages);
        break;
      case 'deepseek':
        yield* this.streamDeepSeek(messages);
        break;
    }
  }

  private async *streamGroq(messages: Message[]): AsyncGenerator<StreamChunk> {
    // Groq uses OpenAI-compatible API
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.modelId,
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 8000,
      }),
    });

    if (!response.ok) {
      throw new Error(`Groq API error: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('No reader available');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim() !== '');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            yield { content: '', done: true };
            return;
          }

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content || '';
            if (content) {
              yield { content, done: false };
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    }
  }

  private async *streamGemini(messages: Message[]): AsyncGenerator<StreamChunk> {
    // Convert messages to Gemini format
    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

    const systemInstruction = messages.find(m => m.role === 'system')?.content;

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${this.modelId}:streamGenerateContent?key=${this.apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents,
          systemInstruction: systemInstruction ? { parts: [{ text: systemInstruction }] } : undefined,
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 8000,
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('No reader available');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim() !== '');

      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          const content = parsed.candidates?.[0]?.content?.parts?.[0]?.text || '';
          if (content) {
            yield { content, done: false };
          }
        } catch (e) {
          // Skip invalid JSON
        }
      }
    }

    yield { content: '', done: true };
  }

  private async *streamDeepSeek(messages: Message[]): AsyncGenerator<StreamChunk> {
    // DeepSeek API endpoint (OpenAI-compatible)
    const response = await fetch('https://api.deepseek.com/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.modelId,
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 8000,
      }),
    });

    if (!response.ok) {
      throw new Error(`DeepSeek API error: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('No reader available');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim() !== '');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            yield { content: '', done: true };
            return;
          }

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content || '';
            if (content) {
              yield { content, done: false };
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    }
  }
}
