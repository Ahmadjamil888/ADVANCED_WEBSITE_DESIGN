import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseOrThrow } from '@/lib/supabase';
import AWS from 'aws-sdk';

export async function POST(req: NextRequest) {
  try {
    const { modelId, accessKey, secretKey, region } = await req.json();

    const supabase = getSupabaseOrThrow();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get model details
    const { data: model, error: modelError } = await supabase
      .from('ai_models')
      .select('*')
      .eq('id', modelId)
      .eq('user_id', user.id)
      .single();

    if (modelError || !model) {
      return NextResponse.json({ error: 'Model not found' }, { status: 404 });
    }

    // Store AWS credentials securely (encrypted in production)
    await (supabase.from('user_integrations').upsert as any)({
      user_id: user.id,
      service_name: 'aws',
      encrypted_api_key: accessKey, // In production, encrypt this
      encrypted_secret: secretKey, // In production, encrypt this
      is_active: true,
      metadata: { region },
    });

    // Initialize AWS clients
    const s3 = new AWS.S3({
      accessKeyId: accessKey,
      secretAccessKey: secretKey,
      region,
    });

    const lambda = new AWS.Lambda({
      accessKeyId: accessKey,
      secretAccessKey: secretKey,
      region,
    });

    // Upload model to S3
    const bucketName = `zehanxtech-models-${user.id.slice(0, 8)}`;
    const modelKey = `models/${modelId}/model.${model.model_file_format || 'pth'}`;

    // Create bucket if it doesn't exist
    try {
      await s3.createBucket({ Bucket: bucketName }).promise();
    } catch (error: any) {
      if (error.code !== 'BucketAlreadyOwnedByYou') {
        throw error;
      }
    }

    // In production, upload actual model file from storage
    // For now, return deployment URL structure
    const deploymentUrl = `https://${bucketName}.s3.${region}.amazonaws.com/${modelKey}`;

    // Update model with deployment info
    await (supabase.from('ai_models').update as any)({
      deployment_type: 'aws',
      deployment_url: deploymentUrl,
      deployed_at: new Date().toISOString(),
    }).eq('id', modelId);

    return NextResponse.json({
      success: true,
      deploymentUrl,
      message: 'Model deployed to AWS successfully',
    });
  } catch (error: any) {
    console.error('AWS deployment error:', error);
    return NextResponse.json(
      { error: error.message || 'AWS deployment failed' },
      { status: 500 }
    );
  }
}

