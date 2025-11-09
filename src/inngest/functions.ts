// src/lib/functions.ts
import { inngest } from "./client";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/lib/supabase";
// E2B sandbox SDK (JS/TS v2)
import { Sandbox, SandboxInfo } from "@e2b/code-interpreter";

// ---------------------------------------------------------------------------
// Supabase setup
// ---------------------------------------------------------------------------
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";
export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_ANON_KEY);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type MessageRole = "user" | "assistant" | "system";

export interface MessageInsert {
  chat_id: string;
  role: MessageRole;
  content: string;
  created_at?: string;
}

export interface ModelConfig {
  type: "nlp" | "cv" | "tabular";
  task: string;
  baseModel: string;
  dataset: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
}

// ---------------------------------------------------------------------------
// Analyze Prompt → Select Model & Dataset
// ---------------------------------------------------------------------------
export function analyzePrompt(prompt: string): ModelConfig {
  const text = (prompt || "").toLowerCase();

  // Vision
  if (/(image|photo|vision|classify images|image classification)/.test(text))
    return {
      type: "cv",
      task: "image-classification",
      baseModel: "google/vit-base-patch16-224",
      dataset: "beans",
    };

  if (/(object detection|detect objects|bounding box|yolo|detr)/.test(text))
    return {
      type: "cv",
      task: "object-detection",
      baseModel: "facebook/detr-resnet-50",
      dataset: "coco",
    };

  // NLP
  if (/(sentiment|emotion)/.test(text))
    return {
      type: "nlp",
      task: "sentiment-analysis",
      baseModel: "distilbert-base-uncased",
      dataset: "imdb",
    };

  if (/(news|topic|classify|text classification)/.test(text))
    return {
      type: "nlp",
      task: "text-classification",
      baseModel: "distilbert-base-uncased",
      dataset: "ag_news",
    };

  if (/(question answering|qa)/.test(text))
    return {
      type: "nlp",
      task: "question-answering",
      baseModel: "distilbert-base-cased-distilled-squad",
      dataset: "squad",
    };

  if (/(summarization|summarise|summarize)/.test(text))
    return {
      type: "nlp",
      task: "summarization",
      baseModel: "facebook/bart-large-cnn",
      dataset: "cnn_dailymail",
    };

  if (/(ner|named entity|entity recognition)/.test(text))
    return {
      type: "nlp",
      task: "token-classification",
      baseModel: "dslim/bert-base-NER",
      dataset: "conll2003",
    };

  if (/(translate|translation)/.test(text))
    return {
      type: "nlp",
      task: "translation",
      baseModel: "Helsinki-NLP/opus-mt-en-de",
      dataset: "wmt14",
    };

  // Tabular
  if (/(csv|tabular|columns|features)/.test(text))
    return {
      type: "tabular",
      task: "tabular-classification",
      baseModel: "xgboost",
      dataset: "adult",
    };

  // Default
  return {
    type: "nlp",
    task: "text-classification",
    baseModel: "distilbert-base-uncased",
    dataset: "ag_news",
  };
}

// ---------------------------------------------------------------------------
// Dataset Selector (Hugging Face → Kaggle fallback)
// ---------------------------------------------------------------------------
export async function findDataset(
  cfg: ModelConfig
): Promise<{ name: string; source: string; url: string | null }> {
  const hfToken = process.env.HF_ACCESS_TOKEN || process.env.HUGGINGFACE_TOKEN;
  const query = cfg.task.replace(/-/g, " ");

  let chosen = { name: cfg.dataset, source: "default", url: null as string | null };

  try {
    const res = await fetch(
      `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}`,
      { headers: hfToken ? { Authorization: `Bearer ${hfToken}` } : {} }
    );

    if (res.ok) {
      const arr = await res.json();
      const match = arr.find((d: any) =>
        (d.id || "").toLowerCase().includes(cfg.dataset.toLowerCase())
      );
      if (match)
        chosen = {
          name: match.id,
          source: "huggingface",
          url: `https://huggingface.co/datasets/${match.id}`,
        };
    }
  } catch {
    // Fallback to Kaggle
  }

  if (!chosen.url) {
    const kaggleMap: Record<string, { name: string; url: string }> = {
      "sentiment-analysis": {
        name: "imdb",
        url: "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
      },
      "text-classification": {
        name: "ag_news",
        url: "https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset",
      },
      "image-classification": {
        name: "cifar-10",
        url: "https://www.kaggle.com/datasets/cifar-10",
      },
      "object-detection": {
        name: "coco",
        url: "https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset",
      },
      "question-answering": {
        name: "squad",
        url: "https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset",
      },
      summarization: {
        name: "cnn_dailymail",
        url: "https://www.kaggle.com/datasets/snapcrack/all-the-news",
      },
      "token-classification": {
        name: "conll2003",
        url: "https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus",
      },
      translation: {
        name: "wmt14",
        url: "https://www.kaggle.com/datasets/yashwantsingh/wmt14-english-german",
      },
      "tabular-classification": {
        name: "adult",
        url: "https://www.kaggle.com/datasets/uciml/adult-census-income",
      },
    };
    if (kaggleMap[cfg.task]) {
      chosen = {
        name: kaggleMap[cfg.task].name,
        source: "kaggle",
        url: kaggleMap[cfg.task].url,
      };
    }
  }

  return chosen;
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
export const genRequirements = (): string =>
  [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "scikit-learn>=1.4.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "gradio>=4.31.0",
    "huggingface_hub>=0.23.0",
    "requests>=2.32.0",
  ].join("\n");

// ---------------------------------------------------------------------------
// Inngest Functions
// ---------------------------------------------------------------------------
export const testFunction = inngest.createFunction(
  { id: "test-function", name: "Test Function" },
  { event: "test/ping" },
  async () => ({ success: true, message: "Inngest working ✅" })
);

export const generateModelCode = inngest.createFunction(
  { id: "ai-workspace-generate-model", name: "Generate + Train + Deploy", concurrency: { limit: 2 } },
  { event: "ai/model.generate" },
  async ({ event, step }) => {
    const { prompt = "", chatId, userId } = event.data as {
      prompt?: string;
      chatId?: string;
      userId?: string;
    };

    // Analyze prompt
    const analysis = await step.run("analyze-prompt", () => analyzePrompt(prompt));

    // Select dataset
    const dataset = await step.run("select-dataset", () => findDataset(analysis));
    const cfg: ModelConfig = { ...analysis, dataset: dataset.name, epochs: 1, batchSize: 8 };

    // Create secure sandbox (v2 API); allow opt-out via env (not recommended)
    const secure = process.env.E2B_SECURE === "false" ? false : true;
    
    // Create sandbox once and reuse the handle
    const sandboxHandle = await step.run("e2b-create-sandbox", async () => {
      const sb = await Sandbox.create({ secure });
      return sb;
    });

    const sandboxId = (sandboxHandle as any).sandboxId || (sandboxHandle as any).id;

    // Write files into sandbox (Python training example)
    await step.run("e2b-write-files", async () => {
      // Reuse the sandbox handle created above
      const files = [
        { path: "/workspace/requirements.txt", data: genRequirements() },
        { path: "/workspace/train.py", data: "import time\nprint('hello from train'); time.sleep(2); print('world')\n" },
      ];
      // Write files using the sandbox handle's files API
      // @ts-ignore - types may vary by version
      await (sandboxHandle as any).files.writeFiles(files);
      return true;
    });

    // Run a background command with stdout streaming and kill after demo
    const runResult = await step.run("e2b-run-bg-command", async () => {
      // Reuse the sandbox handle created above
      const command = await (sandboxHandle as any).commands.run("echo hello; sleep 2; echo world", {
        background: true,
        onStdout: (data: string) => {
          // You can forward logs to an external store if desired
          console.log(`[${sandboxId}]`, data);
        },
      });
      // Demonstrate kill after short delay to avoid lingering process
      setTimeout(() => {
        command.kill().catch(() => void 0);
      }, 3000);
      return { started: true };
    });

    const e2bUrl = `sandbox://${sandboxId}`;

    // Save message
    if (chatId) {
      const msg: MessageInsert = {
        chat_id: chatId,
        role: "assistant",
        content: `✅ Created E2B sandbox and started job for task "${cfg.task}". Sandbox: ${sandboxId}`,
      };
      await (supabase.from as any)("messages").insert(msg as any);
    }

    return { success: true, config: cfg, sandboxId, e2bUrl };
  }
);

// Stub training
export const trainAIModel = inngest.createFunction(
  { id: "ai-train-model", name: "Train AI Model", concurrency: { limit: 1 } },
  { event: "ai/model.train" },
  async ({ event }) => {
    const { modelConfig } = event.data as { modelConfig: ModelConfig };
    return { success: true, received: modelConfig };
  }
);

// Stub deploy
export const deployToE2B = inngest.createFunction(
  { id: "ai-deploy-e2b", name: "Deploy to E2B", concurrency: { limit: 3 } },
  { event: "ai/model.deploy-e2b" },
  async ({ event, step }) => {
    const secure = process.env.E2B_SECURE === "false" ? false : true;

    // Paginated listing example (v2)
    const list = await step.run("e2b-list-sandboxes", async () => {
      const paginator = Sandbox.list({ query: {} });
      const collected: SandboxInfo[] = [];
      while (paginator.hasNext) {
        const items = await paginator.nextItems();
        collected.push(...items);
        if (collected.length >= 50) break; // limit reasonable amount
      }
      return collected.map((s) => ({ id: (s as any).sandboxId || (s as any).id, state: (s as any).state }));
    });

    return { success: true, sandboxes: list };
  }
);
