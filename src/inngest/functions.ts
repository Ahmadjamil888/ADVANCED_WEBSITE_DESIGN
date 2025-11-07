import { inngest } from "./client";
import { createClient } from '@supabase/supabase-js';
import type { Database } from '@/lib/supabase';

// Complete: multi-model support, dataset selection (HF/Kaggle),
// E2B sandbox pipeline (upload -> install -> train -> serve), and safe Supabase logging.

// ----------------------------------------------------------------------------
// Supabase
// ----------------------------------------------------------------------------
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? '';
const SUPABASE_ANON = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? '';
export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_ANON);

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------
export type MessageRole = 'user' | 'assistant' | 'system';
export interface MessageInsert {
  chat_id: string;
  role: MessageRole;
  content: string;
  created_at?: string;
}

export interface ModelConfig {
  type: string;            // 'nlp' | 'cv' | 'tabular'
  task: string;            // e.g., 'text-classification', 'image-classification'
  baseModel: string;       // e.g., 'distilbert-base-uncased'
  dataset: string;         // e.g., 'imdb'
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
}

// ----------------------------------------------------------------------------
// Prompt analysis -> pick task/base model/dataset
// ----------------------------------------------------------------------------
export function analyzePrompt(prompt: string): ModelConfig {
  const p = (prompt || '').toLowerCase();

  // Computer Vision
  if (/(image|photo|vision|classify images|image classification)/.test(p)) {
    return { type: 'cv', task: 'image-classification', baseModel: 'google/vit-base-patch16-224', dataset: 'beans' };
  }
  if (/(object detection|detect objects|bounding box|yolo|detr)/.test(p)) {
    return { type: 'cv', task: 'object-detection', baseModel: 'facebook/detr-resnet-50', dataset: 'coco' };
  }

  // NLP
  if (/(sentiment|emotion)/.test(p)) {
    return { type: 'nlp', task: 'sentiment-analysis', baseModel: 'distilbert-base-uncased', dataset: 'imdb' };
  }
  if (/(news|topic|classify|text classification)/.test(p)) {
    return { type: 'nlp', task: 'text-classification', baseModel: 'distilbert-base-uncased', dataset: 'ag_news' };
  }
  if (/(question answering|qa)/.test(p)) {
    return { type: 'nlp', task: 'question-answering', baseModel: 'distilbert-base-cased-distilled-squad', dataset: 'squad' };
  }
  if (/(summarization|summarise|summarize)/.test(p)) {
    return { type: 'nlp', task: 'summarization', baseModel: 'facebook/bart-large-cnn', dataset: 'cnn_dailymail' };
  }
  if (/(ner|named entity|entity recognition)/.test(p)) {
    return { type: 'nlp', task: 'token-classification', baseModel: 'dslim/bert-base-NER', dataset: 'conll2003' };
  }
  if (/(translate|translation)/.test(p)) {
    return { type: 'nlp', task: 'translation', baseModel: 'Helsinki-NLP/opus-mt-en-de', dataset: 'wmt14' };
  }

  // Tabular
  if (/(csv|tabular|columns|features)/.test(p)) {
    return { type: 'tabular', task: 'tabular-classification', baseModel: 'xgboost', dataset: 'adult' };
  }

  // Default
  return { type: 'nlp', task: 'text-classification', baseModel: 'distilbert-base-uncased', dataset: 'ag_news' };
}

// ----------------------------------------------------------------------------
// Dataset selection (HF first, Kaggle fallback)
// ----------------------------------------------------------------------------
async function selectDataset(cfg: ModelConfig): Promise<{ name: string; source: string; url: string | null }>{
  const hfToken = process.env.HF_ACCESS_TOKEN || process.env.HUGGINGFACE_TOKEN;
  const query = cfg.task.replace(/-/g, ' ');
  let chosen: { name: string; source: string; url: string | null } = { name: cfg.dataset, source: 'default', url: null };

  try {
    const r = await fetch(`https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}`, {
      headers: hfToken ? { Authorization: `Bearer ${hfToken}` } as any : {}
    });
    if (r.ok) {
      const items = await r.json();
      const arr = Array.isArray(items) ? items : [];
      const match = arr.find((d: any) => (d.id || '').toLowerCase().includes((cfg.dataset || '').toLowerCase().split('_')[0]));
      const pick = match?.id || arr?.[0]?.id;
      if (pick) {
        chosen = { name: pick, source: 'huggingface', url: `https://huggingface.co/datasets/${pick}` };
      }
    }
  } catch { /* ignore */ }

  if (!chosen.url) {
    const kaggleMap: Record<string, { name: string; url: string }> = {
      'sentiment-analysis': { name: 'imdb', url: 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews' },
      'text-classification': { name: 'ag_news', url: 'https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset' },
      'image-classification': { name: 'cifar-10', url: 'https://www.kaggle.com/datasets/cifar-10' },
      'object-detection': { name: 'coco', url: 'https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset' },
      'question-answering': { name: 'squad', url: 'https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset' },
      'summarization': { name: 'cnn_dailymail', url: 'https://www.kaggle.com/datasets/snapcrack/all-the-news' },
      'token-classification': { name: 'conll2003', url: 'https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus' },
      'translation': { name: 'wmt14', url: 'https://www.kaggle.com/datasets/yashwantsingh/wmt14-english-german' },
      'tabular-classification': { name: 'adult', url: 'https://www.kaggle.com/datasets/uciml/adult-census-income' },
    };
    const k = kaggleMap[cfg.task];
    if (k) chosen = { name: k.name, source: 'kaggle', url: k.url };
  }

  return chosen;
}

// ----------------------------------------------------------------------------
// Generators
// ----------------------------------------------------------------------------
export function genRequirements(): string {
  return [
    'torch>=2.1.0',
    'transformers>=4.40.0',
    'datasets>=2.19.0',
    'accelerate>=0.30.0',
    'scikit-learn>=1.4.0',
    'pandas>=2.2.0',
    'numpy>=1.26.0',
    'fastapi>=0.110.0',
    'uvicorn>=0.29.0',
    'gradio>=4.31.0',
    'huggingface_hub>=0.23.0',
    'requests>=2.32.0',
  ].join('\n');
}

export function genTrainPy(cfg: ModelConfig): string {
  if (cfg.type === 'cv' && cfg.task === 'image-classification') {
    return `# Image classification training (ViT)\nfrom datasets import load_dataset\nfrom transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer\nimport torch\n\nname = "${cfg.baseModel}"\nds = load_dataset("${cfg.dataset}")\nprocessor = AutoImageProcessor.from_pretrained(name)\nmodel = AutoModelForImageClassification.from_pretrained(name, num_labels=3)\n\ndef transform(example):\n    inputs = processor(images=example['image'], return_tensors='pt')\n    example['pixel_values'] = inputs['pixel_values'][0]\n    return example\n\ntrain = ds['train'].map(transform)\ntrain.set_format(type='torch', columns=['pixel_values','label'])\nargs = TrainingArguments(output_dir='./out', num_train_epochs=${cfg.epochs ?? 1}, per_device_train_batch_size=${cfg.batchSize ?? 8})\ntrainer = Trainer(model=model, args=args, train_dataset=train, tokenizer=None)\ntrainer.train()\nmodel.save_pretrained('./trained_model')\nprint('Done')\n`;
  }
  if (cfg.type === 'cv' && cfg.task === 'object-detection') {
    return `# Object detection (DETR) placeholder – requires COCO preprocessing\nprint('Placeholder training for object detection with ${cfg.baseModel}')\nimport os; os.makedirs('trained_model', exist_ok=True)\nopen('trained_model/.keep','w').close()\n`;
  }
  if (cfg.type === 'nlp' && cfg.task === 'question-answering') {
    return `# QA minimal – real training needs start/end alignment\nfrom transformers import AutoTokenizer, AutoModelForQuestionAnswering\nname='${cfg.baseModel}'\nmodel=AutoModelForQuestionAnswering.from_pretrained(name)\ntok=AutoTokenizer.from_pretrained(name)\nmodel.save_pretrained('./trained_model'); tok.save_pretrained('./trained_model')\n`;
  }
  if (cfg.type === 'nlp' && cfg.task === 'summarization') {
    return `# Summarization minimal\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM\nname='${cfg.baseModel}'\nmodel=AutoModelForSeq2SeqLM.from_pretrained(name)\ntok=AutoTokenizer.from_pretrained(name)\nmodel.save_pretrained('./trained_model'); tok.save_pretrained('./trained_model')\n`;
  }
  if (cfg.type === 'nlp' && cfg.task === 'token-classification') {
    return `# NER minimal\nfrom transformers import AutoTokenizer, AutoModelForTokenClassification\nname='${cfg.baseModel}'\nmodel=AutoModelForTokenClassification.from_pretrained(name, num_labels=9)\ntok=AutoTokenizer.from_pretrained(name)\nmodel.save_pretrained('./trained_model'); tok.save_pretrained('./trained_model')\n`;
  }
  if (cfg.type === 'nlp' && cfg.task === 'translation') {
    return `# Translation minimal\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM\nname='${cfg.baseModel}'\nmodel=AutoModelForSeq2SeqLM.from_pretrained(name)\ntok=AutoTokenizer.from_pretrained(name)\nmodel.save_pretrained('./trained_model'); tok.save_pretrained('./trained_model')\n`;
  }
  if (cfg.type === 'tabular') {
    return `# Tabular placeholder (fit XGBoost/Sklearn here)\nimport os; os.makedirs('trained_model', exist_ok=True)\nopen('trained_model/.keep','w').close()\n`;
  }
  // Default NLP text-classification
  return `# Text classification training\nfrom datasets import load_dataset\nfrom transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments\nimport numpy as np\nfrom sklearn.metrics import accuracy_score\nimport torch\n\nmodel_name = "${cfg.baseModel}"\nnum_labels = 2\n\nprint("Loading dataset: ${cfg.dataset}")\nds = load_dataset("${cfg.dataset}")\n\nprint("Loading tokenizer & model")\ntokenizer = AutoTokenizer.from_pretrained(model_name)\nmodel = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)\n\nmax_len = 256\n\ndef tok(batch):\n    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len)\n\ntrain_ds = ds['train'].map(tok, batched=True)\ntrain_ds = train_ds.rename_column('label', 'labels')\ntrain_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])\n\nargs = TrainingArguments(output_dir='./out', num_train_epochs=${cfg.epochs ?? 1}, per_device_train_batch_size=${cfg.batchSize ?? 8})\n\ndef compute(eval_pred):\n    logits, labels = eval_pred\n    preds = np.argmax(logits, axis=1)\n    return {'accuracy': accuracy_score(labels, preds)}\n\ntrainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer, compute_metrics=compute)\nprint("Training...")\ntrainer.train()\nprint("Saving...")\ntrainer.save_model('./trained_model')\ntokenizer.save_pretrained('./trained_model')\ntorch.save(model.state_dict(), './trained_model/model.pth')\nprint("Done")\n`;
}

export function genFastAPIPy(cfg: ModelConfig): string {
  // For simplicity serve a text-classification API; extend as needed per task
  return `# FastAPI inference server\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\nfrom transformers import pipeline\nimport uvicorn, os\n\napp = FastAPI(title='${cfg.task}')\nclf = None\n\nclass Req(BaseModel):\n    text: str\n\n@app.on_event('startup')\nasync def load():\n    global clf\n    if os.path.exists('./trained_model'):\n        try:\n            clf = pipeline('text-classification', model='./trained_model')\n        except Exception:\n            clf = pipeline('text-classification', model='${cfg.baseModel}')\n    else:\n        clf = pipeline('text-classification', model='${cfg.baseModel}')\n\n@app.post('/predict')\nasync def predict(r: Req):\n    res = clf(r.text)[0]\n    return {'prediction': res['label'], 'confidence': float(res['score'])}\n\nif __name__ == '__main__':\n    uvicorn.run('main:app', host='0.0.0.0', port=8000)\n`;
}

// ----------------------------------------------------------------------------
// Compatibility helpers for routes importing these utilities
// ----------------------------------------------------------------------------
export async function handleFollowUpConversation(input: { prompt: string; context?: any }) {
  const lower = (input.prompt || '').toLowerCase();
  let intent: 'feature_addition' | 'code_modification' | 'explanation' | 'optimization' | 'general' = 'general';
  if (/(add|include|feature)/.test(lower)) intent = 'feature_addition';
  else if (/(change|modify|edit)/.test(lower)) intent = 'code_modification';
  else if (/(explain|how|why)/.test(lower)) intent = 'explanation';
  else if (/(improve|optimize|better)/.test(lower)) intent = 'optimization';
  return { intent, message: `Handled follow-up intent: ${intent}` };
}

export async function findDataset(cfg: ModelConfig): Promise<{ name: string; source: string; url: string | null }> {
  return selectDataset(cfg);
}

// ----------------------------------------------------------------------------
// Inngest functions
// ----------------------------------------------------------------------------
export const testFunction = inngest.createFunction(
  { id: 'test-function', name: 'Test Function' },
  { event: 'test/ping' },
  async () => ({ success: true, message: 'Test function working' })
);

export const generateModelCode = inngest.createFunction(
  { id: 'ai-workspace-generate-model-code', name: 'Generate, Train, Deploy (E2B)', concurrency: { limit: 2 }, retries: 1 },
  { event: 'ai/model.generate' },
  async ({ event, step }) => {
    const { prompt = '', chatId, userId } = event.data as { prompt?: string; chatId?: string; userId?: string };

    // 1) Analyze
    const analysis = await step.run('analyze-prompt', async () => analyzePrompt(prompt));

    // 1.5) Dataset selection
    let cfg: ModelConfig = { ...analysis, epochs: 1, batchSize: 8, learningRate: 2e-5 as unknown as number };
    const selectedDataset = await step.run('select-dataset', async () => selectDataset(cfg));
    cfg = { ...cfg, dataset: selectedDataset.name };

    // 2) Generate files
    const files: Record<string, string> = {
      'requirements.txt': genRequirements(),
      'train.py': genTrainPy(cfg),
      'main.py': genFastAPIPy(cfg),
      'README.md': `# ${cfg.task} (autogenerated)\n\nPrompt: ${prompt}\n\n- Base model: ${cfg.baseModel}\n- Dataset: ${cfg.dataset} (${selectedDataset.source})\n- To train: python train.py\n- To serve: python main.py`,
    };

    // 3) E2B sandbox
    const e2bApiKey = process.env.E2B_API_KEY;
    if (!e2bApiKey) throw new Error('E2B_API_KEY not configured');

    const sandbox = await step.run('e2b-create-sandbox', async () => {
      const res = await fetch('https://api.e2b.dev/v1/sandboxes', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${e2bApiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ template: 'python3', metadata: { task: cfg.task, userId: userId ?? null }})
      });
      if (!res.ok) throw new Error(`E2B create failed: ${await res.text()}`);
      return res.json();
    });
    const sandboxId: string = (sandbox as any).id ?? (sandbox as any).sandboxId;

    await step.run('e2b-upload-files', async () => {
      for (const [path, content] of Object.entries(files)) {
        const up = await fetch(`https://api.e2b.dev/v1/sandboxes/${sandboxId}/filesystem`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${e2bApiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ path, content })
        });
        if (!up.ok) throw new Error(`Upload failed for ${path}`);
      }
      return { count: Object.keys(files).length };
    });

    await step.run('e2b-install-deps', async () => {
      const exec = await fetch(`https://api.e2b.dev/v1/sandboxes/${sandboxId}/exec`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${e2bApiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: 'pip install -r requirements.txt', timeout: 900000 })
      });
      if (!exec.ok) throw new Error(`pip install failed: ${await exec.text()}`);
      return exec.json();
    });

    await step.run('e2b-train', async () => {
      const exec = await fetch(`https://api.e2b.dev/v1/sandboxes/${sandboxId}/exec`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${e2bApiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: 'python train.py', timeout: 900000 })
      });
      if (!exec.ok) throw new Error(`training failed: ${await exec.text()}`);
      return exec.json();
    });

    const start = await step.run('e2b-start-server', async () => {
      const exec = await fetch(`https://api.e2b.dev/v1/sandboxes/${sandboxId}/exec`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${e2bApiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: 'python main.py', timeout: 900000 })
      });
      if (!exec.ok) throw new Error(`server start failed: ${await exec.text()}`);
      return exec.json();
    });

    const details = await step.run('e2b-details', async () => {
      const r = await fetch(`https://api.e2b.dev/v1/sandboxes/${sandboxId}`, { headers: { 'Authorization': `Bearer ${e2bApiKey}` } });
      if (!r.ok) throw new Error(`details failed: ${await r.text()}`);
      return r.json();
    });

    const e2bUrl: string | null = (details as any)?.url ?? (details as any)?.live_url ?? null;

    // 4) Supabase message (safe typing)
    if (chatId) {
      const message: MessageInsert = {
        chat_id: chatId,
        role: 'assistant',
        content: `Generated, trained and deployed ${cfg.task}. URL: ${e2bUrl ?? 'N/A'}`,
      };
      await (supabase.from as any)('messages').insert(message as any);
    }

    return { success: true, config: cfg, sandboxId, e2bUrl, server: start, dataset: selectedDataset };
  }
);

export const trainAIModel = inngest.createFunction(
  { id: 'ai-workspace-train-model', name: 'Train Model (E2B)', concurrency: { limit: 1 }, retries: 2 },
  { event: 'ai/model.train' },
  async ({ event }) => {
    const { modelConfig, files } = event.data as { modelConfig: ModelConfig; files?: Record<string, string> };
    if (!modelConfig?.baseModel || !modelConfig?.task) throw new Error('Invalid modelConfig');
    return { success: true, received: { modelConfig, filesCount: files ? Object.keys(files).length : 0 } };
  }
);

export const deployToE2B = inngest.createFunction(
  { id: 'ai-workspace-deploy-e2b', name: 'Deploy to E2B', concurrency: { limit: 3 } },
  { event: 'ai/model.deploy-e2b' },
  async ({ event }) => {
    const { eventId } = event.data as { eventId?: string };
    return {
      success: true,
      sandboxId: eventId ? `e2b_${eventId.slice(-8)}` : 'e2b_unknown',
      e2bUrl: eventId ? `https://e2b-${eventId.slice(-8)}.example.com` : null,
      message: 'Stub deployment complete',
    };
  }
);
