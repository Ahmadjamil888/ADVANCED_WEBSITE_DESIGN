import { NextRequest, NextResponse } from 'next/server';
import { Sandbox } from '@e2b/code-interpreter';
import { AIClient } from '@/lib/ai/client';
import { AI_MODELS } from '@/lib/ai/models';
import { CODE_AGENT_SYSTEM_PROMPT } from '@/lib/ai/prompts';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

export const runtime = 'nodejs';
export const maxDuration = 300; // 5 minutes

export async function POST(req: NextRequest) {
  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  const sendUpdate = async (type: string, data: any) => {
    await writer.write(
      encoder.encode(`data: ${JSON.stringify({ type, data })}\n\n`)
    );
  };

  (async () => {
    try {
      const { prompt, modelKey, chatId, userId } = await req.json();

      // Get model config
      const model = AI_MODELS[modelKey];
      if (!model) {
        await sendUpdate('error', { message: 'Invalid model selected' });
        await writer.close();
        return;
      }

      await sendUpdate('status', { 
        message: '🤖 Initializing AI agent...', 
        step: 1, 
        total: 7 
      });

      // Step 1: Generate code using AI
      await sendUpdate('status', { 
        message: `💭 Analyzing your request with ${model.name}...`, 
        step: 2, 
        total: 7 
      });

      const aiClient = new AIClient(model.provider, model.id);
      let fullResponse = '';

      for await (const chunk of aiClient.streamCompletion([
        { role: 'system', content: CODE_AGENT_SYSTEM_PROMPT },
        { role: 'user', content: prompt },
      ])) {
        if (!chunk.done) {
          fullResponse += chunk.content;
          await sendUpdate('ai-stream', { content: chunk.content });
        }
      }

      // Step 2: Parse generated files
      await sendUpdate('status', { 
        message: '📝 Extracting generated files...', 
        step: 3, 
        total: 7 
      });

      const files = parseFilesFromResponse(fullResponse);
      
      if (Object.keys(files).length === 0) {
        await sendUpdate('error', { 
          message: 'No files generated. Please try again with a more specific prompt.' 
        });
        await writer.close();
        return;
      }

      await sendUpdate('files', { files: Object.keys(files) });

      // Step 3: Create E2B sandbox
      await sendUpdate('status', { 
        message: '⚡ Creating E2B sandbox environment...', 
        step: 4, 
        total: 7 
      });

      const sandbox = await Sandbox.create('python3');
      await sandbox.setTimeout(1800000); // 30 minutes
      const sandboxId = sandbox.sandboxId;

      await sendUpdate('sandbox', { sandboxId });

      // Step 4: Write files to sandbox
      await sendUpdate('status', { 
        message: '📂 Writing files to sandbox...', 
        step: 5, 
        total: 7 
      });

      for (const [path, content] of Object.entries(files)) {
        await sandbox.files.write(`/home/user/${path}`, content);
        await sendUpdate('file-written', { path });
      }

      // Step 5: Install dependencies
      if (files['requirements.txt']) {
        await sendUpdate('status', { 
          message: '📦 Installing dependencies...', 
          step: 6, 
          total: 7 
        });

        const installResult = await sandbox.commands.run(
          'cd /home/user && pip install -r requirements.txt',
          {
            onStdout: async (data) => {
              await sendUpdate('terminal', { output: data, type: 'stdout' });
            },
            onStderr: async (data) => {
              await sendUpdate('terminal', { output: data, type: 'stderr' });
            },
          }
        );

        if (installResult.exitCode !== 0) {
          await sendUpdate('warning', { 
            message: 'Some dependencies failed to install, but continuing...' 
          });
        }
      }

      // Step 6: Run training
      if (files['train.py']) {
        await sendUpdate('status', { 
          message: '🏋️ Training model... This may take a few minutes.', 
          step: 7, 
          total: 7 
        });

        const trainResult = await sandbox.commands.run(
          'cd /home/user && python train.py',
          {
            onStdout: async (data) => {
              await sendUpdate('training', { output: data });
            },
            onStderr: async (data) => {
              await sendUpdate('training', { output: data, isError: true });
            },
          }
        );

        if (trainResult.exitCode !== 0) {
          await sendUpdate('error', { 
            message: 'Training failed. Check the logs above for details.' 
          });
        }
      }

      // Step 7: Deploy API
      let deploymentUrl = '';
      if (files['app.py']) {
        await sendUpdate('status', { 
          message: '🚀 Deploying FastAPI server...', 
          step: 7, 
          total: 7 
        });

        await sandbox.commands.run(
          'cd /home/user && python -m uvicorn app:app --host 0.0.0.0 --port 8000',
          {
            background: true,
            onStdout: async (data) => {
              await sendUpdate('deployment', { output: data });
            },
          }
        );

        // Wait for server to start
        await new Promise(resolve => setTimeout(resolve, 3000));

        const host = sandbox.getHost(8000);
        deploymentUrl = `http://${host}`;

        await sendUpdate('deployment-url', { url: deploymentUrl });
      }

      // Save to database
      if (chatId && userId) {
        await supabase.from('messages').insert({
          chat_id: chatId,
          role: 'assistant',
          content: fullResponse,
        });
      }

      // Send completion
      await sendUpdate('complete', {
        sandboxId,
        deploymentUrl,
        files: Object.keys(files),
        message: '✅ All done! Your model is trained and deployed.',
      });

    } catch (error: any) {
      await sendUpdate('error', { 
        message: error.message || 'An unexpected error occurred' 
      });
    } finally {
      await writer.close();
    }
  })();

  return new Response(stream.readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

function parseFilesFromResponse(response: string): Record<string, string> {
  const files: Record<string, string> = {};
  const fileRegex = /<file path="([^"]+)">([\s\S]*?)<\/file>/g;
  
  let match;
  while ((match = fileRegex.exec(response)) !== null) {
    const [, path, content] = match;
    files[path] = content.trim();
  }

  // Fallback: try to extract code blocks
  if (Object.keys(files).length === 0) {
    const codeBlockRegex = /```(?:python|txt)?\n([\s\S]*?)```/g;
    let blockIndex = 0;
    const fileNames = ['requirements.txt', 'train.py', 'app.py', 'config.json'];
    
    while ((match = codeBlockRegex.exec(response)) !== null) {
      if (blockIndex < fileNames.length) {
        files[fileNames[blockIndex]] = match[1].trim();
        blockIndex++;
      }
    }
  }

  return files;
}
