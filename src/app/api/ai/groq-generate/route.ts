import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { prompt, model = 'mixtral-8x7b-32768' } = await request.json();

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    if (!process.env.GROQ_API_KEY) {
      return NextResponse.json(
        { error: 'GROQ_API_KEY is not configured' },
        { status: 500 }
      );
    }

    console.log('[groq-generate] Generating code with model:', model);
    console.log('[groq-generate] Prompt:', prompt);

    // Dynamic import to handle groq-sdk
    const { default: Groq } = await import('groq-sdk');
    
    const groq: any = new Groq({
      apiKey: process.env.GROQ_API_KEY,
    });

    const systemPrompt = `You are an expert AI model architect and Python developer. Your task is to:
1. Analyze the user's request for an AI model
2. Generate complete, production-ready Python code for training a PyTorch model
3. Include dataset finding/creation logic
4. Include model architecture, training loop, and evaluation
5. Ensure the code is self-contained and can run in a sandbox environment

Format your response as follows:
<code>
[Complete Python code here]
</code>

<dataset>
[Description of dataset and how to obtain it]
</dataset>

<model_type>
[Type of model: classification, regression, nlp, vision, etc.]
</model_type>

<requirements>
[List of pip packages needed]
</requirements>`;

    const message = await groq.messages.create({
      model: model,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    const responseText =
      message.content[0].type === 'text' ? message.content[0].text : '';

    // Parse the response
    const codeMatch = responseText.match(/<code>([\s\S]*?)<\/code>/);
    const datasetMatch = responseText.match(/<dataset>([\s\S]*?)<\/dataset>/);
    const modelTypeMatch = responseText.match(/<model_type>([\s\S]*?)<\/model_type>/);
    const requirementsMatch = responseText.match(/<requirements>([\s\S]*?)<\/requirements>/);

    const code = codeMatch ? codeMatch[1].trim() : '';
    const dataset = datasetMatch ? datasetMatch[1].trim() : '';
    const modelType = modelTypeMatch ? modelTypeMatch[1].trim() : 'custom';
    const requirements = requirementsMatch
      ? requirementsMatch[1]
          .split('\n')
          .map((r: string) => r.trim())
          .filter((r: string) => r && !r.startsWith('#'))
      : ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn'];

    console.log('[groq-generate] Generated code length:', code.length);
    console.log('[groq-generate] Model type:', modelType);
    console.log('[groq-generate] Requirements:', requirements);

    return NextResponse.json({
      success: true,
      code,
      dataset,
      modelType,
      requirements,
      model,
      rawResponse: responseText,
    });
  } catch (error) {
    console.error('[groq-generate] Error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to generate code',
      },
      { status: 500 }
    );
  }
}
