// E2B Sandbox Integration
// Following official E2B documentation

import { Sandbox } from '@e2b/code-interpreter';

export interface E2BExecutionResult {
  text: string;
  results: any[];
  logs: {
    stdout: string[];
    stderr: string[];
  };
  error?: any;
}

export class E2BManager {
  private sandbox: Sandbox | null = null;
  private sandboxId: string | null = null;
  private accessToken: string | null = null;

  /**
   * Create a new E2B sandbox
   * Following: https://e2b.dev/docs
   */
  async createSandbox(): Promise<string> {
    try {
      const secureEnv = (process.env.E2B_SECURE ?? 'true').toLowerCase();
      const secure = secureEnv === 'true' || secureEnv === '1';
      const timeoutMs = Number(process.env.E2B_TIMEOUT_MS ?? 1800000);
      this.sandbox = await Sandbox.create({ secure, timeoutMs });
      // Access token is available in SDK v2 for secure access
      // @ts-ignore - token may be internal/private in types
      this.accessToken = (this.sandbox as any)?.accessToken ?? null;
      this.sandboxId = this.sandbox.sandboxId;
      console.log(`✅ E2B Sandbox created: ${this.sandboxId}`);
      if (this.accessToken) {
        console.log('🔑 Secure access token obtained for sandbox controller');
      } else {
        console.log('⚠️ No secure access token available (secure access may be disabled)');
      }
      return this.sandboxId;
    } catch (error) {
      console.error('❌ Failed to create E2B sandbox:', error);
      throw new Error(`E2B sandbox creation failed: ${error}`);
    }
  }

  /**
   * Connect to an existing sandbox
   */
  async connectToSandbox(sandboxId: string): Promise<string> {
    try {
      console.log(`🔌 Connecting to existing sandbox: ${sandboxId}`);
      this.sandbox = await Sandbox.connect(sandboxId);
      this.sandboxId = sandboxId;
      console.log(`✅ Connected to sandbox: ${this.sandboxId}`);
      return this.sandboxId;
    } catch (error) {
      console.error('❌ Failed to connect to sandbox:', error);
      throw new Error(`Sandbox connection failed: ${error}`);
    }
  }

  /**
   * Get or create sandbox - reuse if available
   */
  async getOrCreateSandbox(existingSandboxId?: string): Promise<string> {
    if (existingSandboxId) {
      try {
        return await this.connectToSandbox(existingSandboxId);
      } catch (error) {
        console.warn('⚠️ Failed to connect to existing sandbox, creating new one');
      }
    }
    return await this.createSandbox();
  }

  /**
   * Write a file to the sandbox
   * Following: https://e2b.dev/docs/sandbox/files
   */
  async writeFile(path: string, content: string): Promise<void> {
    if (!this.sandbox) {
      throw new Error('Sandbox not initialized');
    }

    try {
      await this.sandbox.files.write(path, content);
      console.log(`✅ File written: ${path}`);
    } catch (error) {
      console.error(`❌ Failed to write file ${path}:`, error);
      throw error;
    }
  }

  /**
   * Write multiple files to the sandbox
   */
  async writeFiles(files: Record<string, string>): Promise<void> {
    for (const [path, content] of Object.entries(files)) {
      await this.writeFile(`/home/user/${path}`, content);
    }
  }

  /**
   * Run a command in the sandbox
   * Following: https://e2b.dev/docs/sandbox/commands
   */
  async runCommand(
    command: string,
    onStdout?: (data: string) => void,
    onStderr?: (data: string) => void
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    if (!this.sandbox) {
      throw new Error('Sandbox not initialized');
    }

    try {
      const buffers = { stdout: '', stderr: '' };

      const result = await this.sandbox.commands.run(command, {
        onStdout: (data: string) => {
          buffers.stdout += data;
          if (onStdout) onStdout(data);
        },
        onStderr: (data: string) => {
          buffers.stderr += data;
          if (onStderr) onStderr(data);
        },
      });

      return {
        exitCode: result.exitCode,
        stdout: buffers.stdout,
        stderr: buffers.stderr,
      };
    } catch (error) {
      console.error(`❌ Command failed: ${command}`, error);
      throw error;
    }
  }

  /**
   * Run Python code in the sandbox
   * Following: https://e2b.dev/docs/code-interpreter
   */
  async runCode(code: string): Promise<E2BExecutionResult> {
    if (!this.sandbox) {
      throw new Error('Sandbox not initialized');
    }

    try {
      const execution = await this.sandbox.runCode(code);
      
      return {
        text: execution.text || '',
        results: execution.results || [],
        logs: execution.logs || { stdout: [], stderr: [] },
        error: execution.error,
      };
    } catch (error) {
      console.error('❌ Code execution failed:', error);
      throw error;
    }
  }

  /**
   * Get sandbox host for port forwarding
   * Following: https://e2b.dev/docs/sandbox/networking
   */
  getHost(port: number = 8000): string {
    if (!this.sandbox) {
      throw new Error('Sandbox not initialized');
    }

    const host = this.sandbox.getHost(port);
    if (!host) {
      throw new Error('Failed to get sandbox host');
    }
    return `https://${host}`;
  }

  /**
   * Install Python dependencies
   */
  async installDependencies(requirementsPath: string = '/home/user/requirements.txt'): Promise<void> {
    console.log('📦 Installing dependencies...');
    
    const result = await this.runCommand(
      `pip install -r ${requirementsPath}`,
      (data) => console.log(`[pip] ${data}`),
      (data) => console.error(`[pip error] ${data}`)
    );

    if (result.exitCode !== 0) {
      console.warn('⚠️ Some dependencies failed to install, continuing...');
    } else {
      console.log('✅ Dependencies installed successfully');
    }
  }

  /**
   * Run training script
   */
  async runTraining(scriptPath: string = '/home/user/train.py'): Promise<void> {
    console.log('🏋️ Starting training...');
    
    const result = await this.runCommand(
      `python ${scriptPath}`,
      (data) => console.log(`[training] ${data}`),
      (data) => console.error(`[training error] ${data}`)
    );

    if (result.exitCode !== 0) {
      throw new Error(`Training failed with exit code ${result.exitCode}`);
    }

    console.log('✅ Training completed successfully');
  }

  /**
   * Deploy FastAPI server in background
   * Following: https://e2b.dev/docs/sandbox/commands#running-commands-in-background
   */
  async deployAPI(
    appPath: string = '/home/user/app.py',
    port: number = 8000,
    opts?: {
      startCommand?: string;
      fallbackStartCommand?: string;
      cwd?: string;
      waitSeconds?: number;
    }
  ): Promise<string> {
    if (!this.sandbox) {
      throw new Error('Sandbox not initialized');
    }

    console.log('🚀 Deploying FastAPI server...');
    
    try {
      // Start server in background - E2B will keep it running
      const startCmd = opts?.startCommand
        ? opts.startCommand
        : `cd /home/user && python -m uvicorn app:app --host 0.0.0.0 --port ${port}`;
      
      // Run in background using E2B's background option
      await this.sandbox.commands.run(startCmd, {
        background: true,
        onStdout: (data) => console.log(`[uvicorn] ${data}`),
        onStderr: (data) => console.log(`[uvicorn] ${data}`),
      });
      
      console.log('✅ Uvicorn started in background');

      // Wait for server to start with retries
      const totalWait = Math.max(5, opts?.waitSeconds ?? 30);
      let ready = false;
      for (let i = 0; i < totalWait; i++) {
        const checkCmd = `curl -s -o /dev/null -w "%{http_code}" http://localhost:${port}/ || echo "000"`;
        const checkResult = await this.sandbox.commands.run(checkCmd);
        if (checkResult.stdout.trim() !== '000') {
          ready = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Fallback attempt if not ready
      if (!ready && opts?.fallbackStartCommand) {
        console.warn('⚠️ Primary server start did not respond, attempting fallback start command');
        await this.sandbox.commands.run(opts.fallbackStartCommand, {
          background: true,
          onStdout: (data) => console.log(`[server] ${data}`),
          onStderr: (data) => console.log(`[server] ${data}`),
        });
        for (let i = 0; i < 20; i++) {
          const checkCmd = `curl -s -o /dev/null -w "%{http_code}" http://localhost:${port}/ || echo "000"`;
          const checkResult = await this.sandbox.commands.run(checkCmd);
          if (checkResult.stdout.trim() !== '000') {
            ready = true;
            break;
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }

      if (!ready) {
        console.warn('⚠️ Server not responding on health check; exposing URL anyway');
      } else {
        console.log('✅ Server is responding');
      }

      const url = this.getHost(port);
      console.log(`✅ API deployed at: ${url}`);
      
      return url;
    } catch (error: any) {
      console.error('❌ Failed to deploy API:', error);
      throw new Error(`API deployment failed: ${error.message}`);
    }
  }

  /**
   * Get sandbox ID
   */
  getSandboxId(): string | null {
    return this.sandboxId;
  }

  /**
   * Close the sandbox (cleanup)
   */
  async close(): Promise<void> {
    if (this.sandbox) {
      // E2B sandboxes auto-close after timeout
      // Manual cleanup not needed
      this.sandbox = null;
      console.log('🔒 Sandbox reference cleared');
    }
  }
}

/**
 * Helper function to create and manage E2B sandbox
 * Usage:
 * 
 * const manager = new E2BManager();
 * await manager.createSandbox();
 * await manager.writeFiles(files);
 * await manager.installDependencies();
 * await manager.runTraining();
 * const url = await manager.deployAPI();
 * await manager.close();
 */
export async function createE2BSandbox(): Promise<E2BManager> {
  const manager = new E2BManager();
  await manager.createSandbox();
  return manager;
}
