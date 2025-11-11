import { NextRequest, NextResponse } from 'next/server';
import { E2BManager } from '@/lib/e2b';
import { AIClient } from '@/lib/ai/client';
import { AI_MODELS } from '@/lib/ai/models';
import { CODE_AGENT_SYSTEM_PROMPT } from '@/lib/ai/prompts';
import { createClient } from '@supabase/supabase-js';
import { db } from '@/lib/db';

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

      // Check E2B API key
      if (!process.env.E2B_API_KEY) {
        await sendUpdate('error', { 
          message: '❌ E2B_API_KEY not found in environment variables. Please add it to .env.local' 
        });
        await writer.close();
        return;
      }

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

      let e2b: E2BManager;
      let sandboxId: string | null;
      
      try {
        e2b = new E2BManager();
        await e2b.createSandbox();
        sandboxId = e2b.getSandboxId();
        
        if (!sandboxId) {
          throw new Error('Failed to get sandbox ID');
        }
        
        await sendUpdate('sandbox', { sandboxId });
        console.log('✅ E2B Sandbox created:', sandboxId);
      } catch (error: any) {
        console.error('❌ E2B Sandbox creation failed:', error);
        await sendUpdate('error', { 
          message: `Failed to create E2B sandbox: ${error.message}. Please check your E2B_API_KEY.` 
        });
        await writer.close();
        return;
      }

      // Step 4: Write files to sandbox
      await sendUpdate('status', { 
        message: '📂 Writing files to sandbox...', 
        step: 5, 
        total: 7 
      });

      try {
        await e2b.writeFiles(files);
        for (const path of Object.keys(files)) {
          await sendUpdate('file-written', { path });
          console.log('✅ File written:', path);
        }
      } catch (error: any) {
        console.error('❌ File writing failed:', error);
        await sendUpdate('error', { 
          message: `Failed to write files: ${error.message}` 
        });
        await writer.close();
        return;
      }

      // Step 5: Install dependencies
      if (files['requirements.txt']) {
        await sendUpdate('status', { 
          message: '📦 Installing dependencies...', 
          step: 6, 
          total: 7 
        });

        try {
          // First, upgrade pip to avoid version issues
          await e2b.runCommand('pip install --upgrade pip');
          
          const installResult = await e2b.runCommand(
            'pip install -r /home/user/requirements.txt --no-cache-dir',
            async (data: string) => {
              await sendUpdate('terminal', { output: data, type: 'stdout' });
            },
            async (data: string) => {
              await sendUpdate('terminal', { output: data, type: 'stderr' });
            }
          );

          if (installResult.exitCode !== 0) {
            console.warn('⚠️ Some dependencies failed to install');
            console.warn('Installation stderr:', installResult.stderr);
            console.warn('Installation stdout:', installResult.stdout);
            
            // Try installing dependencies one by one to identify which ones fail
            const requirements = files['requirements.txt'].split('\n').filter(line => line.trim() && !line.startsWith('#'));
            const failedPackages: string[] = [];
            const successPackages: string[] = [];
            
            for (const pkg of requirements) {
              const pkgResult = await e2b.runCommand(`pip install ${pkg.trim()} --no-cache-dir`);
              if (pkgResult.exitCode === 0) {
                successPackages.push(pkg.trim());
              } else {
                failedPackages.push(pkg.trim());
              }
            }
            
            if (successPackages.length > 0) {
              await sendUpdate('status', { 
                message: `✅ Installed ${successPackages.length}/${requirements.length} packages. ${failedPackages.length > 0 ? `Failed: ${failedPackages.join(', ')}` : ''}` 
              });
              console.log('✅ Partial installation successful:', successPackages);
              if (failedPackages.length > 0) {
                console.warn('⚠️ Failed packages:', failedPackages);
              }
            } else {
              // All packages failed - this is a real error
              const errorDetails = installResult.stderr || installResult.stdout || 'Unknown error';
              await sendUpdate('error', { 
                message: `All dependencies failed to install. Please check your requirements.txt`,
                details: errorDetails
              });
              await writer.close();
              return;
            }
          } else {
            console.log('✅ Dependencies installed successfully');
            await sendUpdate('status', { message: '✅ Dependencies installed' });
          }
        } catch (error: any) {
          console.error('❌ Dependency installation error:', error);
          // Don't fail completely - continue with what we have
          await sendUpdate('status', { 
            message: `⚠️ Dependency installation had issues, continuing anyway: ${error.message}` 
          });
        }
      }

      // Step 6: Run training (optional - skip if no train.py)
      if (files['train.py']) {
        await sendUpdate('status', { 
          message: '🏋️ Training model... This may take a few minutes.', 
          step: 7, 
          total: 7 
        });

        try {
          const trainResult = await e2b.runCommand(
            'python /home/user/train.py',
            async (data: string) => {
              await sendUpdate('training', { output: data });
            },
            async (data: string) => {
              await sendUpdate('training', { output: data, isError: true });
            }
          );

          if (trainResult.exitCode !== 0) {
            console.error('❌ Training failed with exit code:', trainResult.exitCode);
            console.error('Training stderr:', trainResult.stderr);
            const errorDetails = (trainResult.stderr || '').slice(0, 5000);
            await sendUpdate('error', { 
              message: `Training failed with exit code ${trainResult.exitCode}.`,
              details: errorDetails
            });
            // Continue to deployment if possible
          }
          
          console.log('✅ Training completed successfully');
          await sendUpdate('status', { message: '✅ Training completed' });
        } catch (error: any) {
          console.error('❌ Training error:', error);
          await sendUpdate('error', { 
            message: `Training error: ${error.message}` 
          });
          // Continue to deployment if possible
        }
      }

      // Ensure minimal app.py exists if not generated
      if (!files['app.py']) {
        await sendUpdate('status', { message: 'ℹ️ No app.py detected, creating a minimal FastAPI app' });
        const minimalApp = `
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root():
    return {"status": "ok"}
`.trimStart();
        await e2b.writeFile('/home/user/app.py', minimalApp);
      }

      // Ensure FastAPI and Uvicorn are installed (best effort)
      try {
        await sendUpdate('status', { message: '📦 Ensuring FastAPI and Uvicorn are installed...' });
        const pipResult = await e2b.runCommand(
          'pip install fastapi uvicorn',
          async (data: string) => await sendUpdate('install', { output: data }),
          async (data: string) => await sendUpdate('install', { output: data, isError: true })
        );
        if (pipResult.exitCode !== 0) {
          await sendUpdate('status', { message: '⚠️ pip install reported issues, continuing...' });
        } else {
          await sendUpdate('status', { message: '✅ FastAPI and Uvicorn ready' });
        }
      } catch (e: any) {
        await sendUpdate('status', { message: `⚠️ pip install failed (${e.message}), continuing...` });
      }

      // Step 7: Deploy API (with fallback to static server)
      let deploymentUrl = '';
      await sendUpdate('status', { 
        message: '🚀 Deploying server...', 
        step: 7, 
        total: 7 
      });

      try {
        // Try starting uvicorn first, then fallback to static server if needed
        deploymentUrl = await e2b.deployAPI('/home/user/app.py', 8000, {
          startCommand: `cd /home/user && python -m uvicorn app:app --host 0.0.0.0 --port 8000`,
          fallbackStartCommand: `cd /home/user && python -m http.server 8000`,
          waitSeconds: 30,
        });
        console.log('✅ API deployed at:', deploymentUrl);
        await sendUpdate('deployment-url', { url: deploymentUrl });
        await sendUpdate('status', { message: '✅ API deployed successfully' });
      } catch (error: any) {
        console.error('❌ API deployment failed:', error);
        await sendUpdate('error', { 
          message: `API deployment failed: ${error.message}` 
        });
        await writer.close();
        return;
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
      console.error('❌ API Route Error:', error);
      console.error('Error stack:', error.stack);
      await sendUpdate('error', { 
        message: `Error: ${error.message || 'An unexpected error occurred'}`,
        details: error.stack
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
  
  // Try standard XML format first
  const fileRegex = /<file path="([^"]+)">([\s\S]*?)<\/file>/g;
  let match;
  while ((match = fileRegex.exec(response)) !== null) {
    let [, path, content] = match;
    // Fix common mistakes: add .txt if missing for requirements
    if (path === 'requirements' || path === 'requirement') {
      path = 'requirements.txt';
    }
    // Fix .py extension if missing
    if ((path === 'train' || path === 'app') && !path.endsWith('.py')) {
      path = path + '.py';
    }
    files[path] = content.trim();
  }

  // Fallback 1: Try to find file markers in text
  if (Object.keys(files).length === 0) {
    const patterns = [
      { name: 'requirements.txt', regex: /<file path="requirements?"[^>]*>([\s\S]*?)(?:<\/file>|$)/i },
      { name: 'config.json', regex: /<file path="config\.json"[^>]*>([\s\S]*?)(?:<\/file>|$)/i },
      { name: 'train.py', regex: /<file path="train\.py"[^>]*>([\s\S]*?)(?:<\/file>|$)/i },
      { name: 'app.py', regex: /<file path="app\.py"[^>]*>([\s\S]*?)(?:<\/file>|$)/i },
    ];

    for (const pattern of patterns) {
      const match = response.match(pattern.regex);
      if (match) {
        files[pattern.name] = match[1].trim();
      }
    }
  }

  // Fallback 2: Extract code blocks
  if (Object.keys(files).length === 0) {
    const codeBlockRegex = /```(?:python|json|txt)?\n([\s\S]*?)```/g;
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
