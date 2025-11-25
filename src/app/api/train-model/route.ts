import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';
import { Sandbox } from '@e2b/code-interpreter';
import { createClient } from '@supabase/supabase-js';
import { getAuth } from '@clerk/nextjs/server';

interface TrainingRequest {
  name: string;
  description: string;
  modelType: 'transformer' | 'lstm' | 'cnn' | 'custom';
  datasetSource: 'firecrawl' | 'huggingface' | 'kaggle' | 'github';
  githubRepo?: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

interface TrainingStats {
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: string;
}

// Initialize clients
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''
);

export async function POST(request: NextRequest) {
  try {
    const body: TrainingRequest = await request.json();

    // Validate required fields
    if (!body.name || !body.modelType) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Get user ID from auth
    const { userId } = getAuth(request);
    if (!userId) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Create a readable stream for SSE
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        let trainingJobId: string | null = null;
        let modelId: string | null = null;
        let allStats: TrainingStats[] = [];

        try {
          // Send initial message
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'start',
                message: 'Starting model training...',
              })}\n\n`
            )
          );

          // Step 1: Create training job in Supabase
          const { data: jobData, error: jobError } = await supabase
            .from('training_jobs')
            .insert({
              user_id: userId,
              status: 'initializing',
              progress: 0,
            })
            .select()
            .single();

          if (jobError) throw new Error(`Failed to create training job: ${jobError.message}`);
          trainingJobId = jobData?.id;

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'status',
                message: 'Fetching dataset...',
                progress: 10,
              })}\n\n`
            )
          );

          // Step 2: Fetch dataset using Firecrawl
          let datasetContent = '';
          if (body.datasetSource === 'firecrawl') {
            datasetContent = await fetchFirecrawlDataset(body.description);
          } else if (body.datasetSource === 'github') {
            datasetContent = await fetchGithubDataset(body.githubRepo || '');
          } else if (body.datasetSource === 'huggingface') {
            datasetContent = await fetchHuggingFaceDataset(body.description);
          } else {
            datasetContent = 'Sample training data for model generation';
          }

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'status',
                message: 'Generating training code with Groq...',
                progress: 20,
              })}\n\n`
            )
          );

          // Step 3: Generate training code using Groq
          const trainingCode = await generateTrainingCodeWithGroq(
            body,
            datasetContent
          );

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'status',
                message: 'Creating E2B sandbox...',
                progress: 30,
              })}\n\n`
            )
          );

          // Step 4: Create E2B sandbox and execute training
          let sandbox: Sandbox | null = null;
          try {
            sandbox = await Sandbox.create();

            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'status',
                  message: 'Executing training in E2B sandbox...',
                  progress: 40,
                  sandboxUrl: sandbox.getMetadata().sandboxUrl,
                })}\n\n`
              )
            );

            // Execute training code
            const result = await sandbox.runCode(trainingCode);

            // Parse training stats from output
            if (result.stdout) {
              const lines = result.stdout.split('\n');
              for (const line of lines) {
                if (line.startsWith('STATS:')) {
                  try {
                    const statsStr = line.replace('STATS:', '').trim();
                    const stats = JSON.parse(statsStr);
                    allStats.push(stats);

                    // Send stats to client
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify({
                          type: 'stats',
                          stats: stats,
                          progress: 40 + (allStats.length / body.epochs) * 50,
                        })}\n\n`
                      )
                    );

                    // Update training job progress
                    if (trainingJobId) {
                      await supabase
                        .from('training_jobs')
                        .update({
                          progress: Math.min(90, 40 + (allStats.length / body.epochs) * 50),
                          current_epoch: stats.epoch,
                          total_epochs: body.epochs,
                        })
                        .eq('id', trainingJobId);
                    }
                  } catch (e) {
                    console.error('Failed to parse stats:', e);
                  }
                }
              }
            }

            // Get model file from sandbox
            let modelData = '';
            try {
              const modelFile = await sandbox.downloadFile(`/tmp/${body.name}.pt`);
              modelData = Buffer.from(modelFile).toString('base64');
            } catch (e) {
              console.error('Failed to download model from sandbox:', e);
            }

            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'status',
                  message: 'Saving model to Supabase...',
                  progress: 90,
                })}\n\n`
              )
            );

            // Step 5: Save model metadata to Supabase
            const finalStats = allStats[allStats.length - 1] || {
              epoch: body.epochs,
              loss: 0,
              accuracy: 0,
              timestamp: new Date().toISOString(),
            };

            const { data: modelData_response, error: modelError } = await supabase
              .from('trained_models')
              .insert({
                user_id: userId,
                name: body.name,
                description: body.description,
                model_type: body.modelType,
                dataset_source: body.datasetSource,
                final_loss: finalStats.loss,
                final_accuracy: finalStats.accuracy,
                epochs_trained: body.epochs,
                model_path: `/models/${body.name}.pt`,
                stats: allStats,
                sandbox_url: sandbox?.getMetadata().sandboxUrl || null,
              })
              .select()
              .single();

            if (modelError) throw new Error(`Failed to save model: ${modelError.message}`);
            modelId = modelData_response?.id;

            // Update training job to complete
            if (trainingJobId) {
              await supabase
                .from('training_jobs')
                .update({
                  status: 'completed',
                  progress: 100,
                  model_id: modelId,
                })
                .eq('id', trainingJobId);
            }

            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'complete',
                  message: 'Training completed successfully',
                  modelId: modelId,
                  stats: finalStats,
                  sandboxUrl: sandbox?.getMetadata().sandboxUrl,
                  progress: 100,
                })}\n\n`
              )
            );
          } finally {
            // Cleanup sandbox
            if (sandbox) {
              try {
                await sandbox.kill();
              } catch (e) {
                console.error('Failed to cleanup sandbox:', e);
              }
            }
          }

          controller.close();
        } catch (error) {
          console.error('Training error:', error);

          // Update training job with error
          if (trainingJobId) {
            await supabase
              .from('training_jobs')
              .update({
                status: 'failed',
              })
              .eq('id', trainingJobId);
          }

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: 'error',
                message: error instanceof Error ? error.message : 'Training failed',
              })}\n\n`
            )
          );
          controller.close();
        }
      },
    });

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Request error:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}

async function generateTrainingCodeWithGroq(
  config: TrainingRequest,
  datasetContent: string
): Promise<string> {
  const prompt = `Generate a complete PyTorch training script for the following:
- Model Type: ${config.modelType}
- Description: ${config.description}
- Epochs: ${config.epochs}
- Batch Size: ${config.batchSize}
- Learning Rate: ${config.learningRate}

Dataset sample:
${datasetContent.substring(0, 500)}...

Requirements:
1. Create a ${config.modelType} model architecture
2. Load and preprocess the dataset
3. Train for ${config.epochs} epochs
4. Print stats in format: STATS:{"epoch":X,"loss":Y,"accuracy":Z,"timestamp":"..."}
5. Save model to /tmp/${config.name}.pt
6. Use PyTorch and standard libraries only
7. Make it production-ready with error handling

Return ONLY the Python code, no explanations.`;

  const message = await groq.chat.completions.create({
    model: 'mixtral-8x7b-32768',
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
    temperature: 0.7,
    max_tokens: 2000,
  });

  let code = message.choices[0]?.message?.content || '';

  // Clean up code if it has markdown formatting
  if (code.includes('```python')) {
    code = code.split('```python')[1].split('```')[0];
  } else if (code.includes('```')) {
    code = code.split('```')[1].split('```')[0];
  }

  return code.trim();
}

async function fetchFirecrawlDataset(description: string): Promise<string> {
  const firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
  if (!firecrawlApiKey) {
    throw new Error('FIRECRAWL_API_KEY not configured');
  }

  try {
    // Search for relevant Wikipedia articles
    const searchTerms = description.split(' ').slice(0, 3).join(' ');

    const response = await fetch('https://api.firecrawl.dev/v1/scrape', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${firecrawlApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: `https://en.wikipedia.org/wiki/${searchTerms.replace(/ /g, '_')}`,
        formats: ['markdown'],
      }),
    });

    if (response.ok) {
      const data = await response.json();
      return data.markdown || description;
    }

    return description;
  } catch (error) {
    console.error('Firecrawl fetch error:', error);
    return description;
  }
}

async function fetchGithubDataset(repoUrl: string): Promise<string> {
  try {
    // Extract repo info and fetch README
    const match = repoUrl.match(/github\.com\/([^/]+)\/([^/]+)/);
    if (!match) throw new Error('Invalid GitHub URL');

    const [, owner, repo] = match;
    const readmeUrl = `https://raw.githubusercontent.com/${owner}/${repo}/main/README.md`;

    const response = await fetch(readmeUrl);
    if (response.ok) {
      return await response.text();
    }

    return `GitHub repository: ${repoUrl}`;
  } catch (error) {
    console.error('GitHub fetch error:', error);
    return `GitHub repository: ${repoUrl}`;
  }
}

async function fetchHuggingFaceDataset(description: string): Promise<string> {
  try {
    // This would integrate with Hugging Face API
    // For now, return a placeholder
    return `Hugging Face dataset for: ${description}`;
  } catch (error) {
    console.error('Hugging Face fetch error:', error);
    return description;
  }
}
