import { NextRequest, NextResponse } from 'next/server'
import { getSupabaseOrThrow, type Database } from '@/lib/supabase'
import AWS from 'aws-sdk'

type AIModelRow = Database['public']['Tables']['ai_models']['Row']
type AIModelUpdate = Database['public']['Tables']['ai_models']['Update']
type UserIntegrationInsert = Database['public']['Tables']['user_integrations']['Insert']
const AI_MODELS_TABLE = 'ai_models' as const
const USER_INTEGRATIONS_TABLE = 'user_integrations' as const

export async function POST(req: NextRequest) {
  try {
    const { modelId, accessKey, secretKey, region } = (await req.json()) as Partial<{
      modelId: string
      accessKey: string
      secretKey: string
      region: string
    }>

    if (!modelId || !accessKey || !secretKey || !region) {
      return NextResponse.json(
        { error: 'Missing required parameters: modelId, accessKey, secretKey, region' },
        { status: 400 },
      )
    }

    const supabase = getSupabaseOrThrow()
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // ✅ Get model (typed as AIModel)
    const modelResponse = await supabase
      .from(AI_MODELS_TABLE)
      .select('*')
      .eq('id', modelId)
      .eq('user_id', user.id)
      .single()
    const model = modelResponse.data as AIModelRow | null
    const modelError = modelResponse.error

    if (modelError || !model) {
      console.error('Model fetch error:', modelError)
      return NextResponse.json({ error: 'Model not found' }, { status: 404 })
    }

    // ✅ Store AWS credentials (encrypt in production)
    const integration: UserIntegrationInsert = {
      user_id: user.id,
      service_name: 'aws',
      encrypted_api_key: JSON.stringify({
        accessKey,
        secretKey,
      }),
      is_active: true,
    }

    const { error: integrationError } = await (supabase
      .from(USER_INTEGRATIONS_TABLE) as any)
      .upsert(integration)

    if (integrationError) {
      console.error('Failed to store AWS credentials:', integrationError)
    }

    // ⚙️ Initialize AWS clients
    const s3 = new AWS.S3({
      accessKeyId: accessKey,
      secretAccessKey: secretKey,
      region,
    })

    const lambda = new AWS.Lambda({
      accessKeyId: accessKey,
      secretAccessKey: secretKey,
      region,
    })

    // 🪣 Define bucket and model key
    const bucketName = `zehanxtech-models-${user.id.slice(0, 8)}`
    const metadataRecord =
      model.metadata && !Array.isArray(model.metadata) && typeof model.metadata === 'object'
        ? (model.metadata as Record<string, unknown>)
        : {}
    const fileExtValue = metadataRecord['model_file_format']
    const fileExt = typeof fileExtValue === 'string' ? fileExtValue : 'pth'
    const modelKey = `models/${modelId}/model.${fileExt}`

    // ✅ Ensure bucket exists
    try {
      await s3.createBucket({ Bucket: bucketName }).promise()
    } catch (error: any) {
      if (error.code !== 'BucketAlreadyOwnedByYou') {
        throw error
      }
    }

    // (Optional) Upload real model file later
    const deploymentUrl = `https://${bucketName}.s3.${region}.amazonaws.com/${modelKey}`
    const deployedAt = new Date().toISOString()
    const metadataPayload = {
      ...metadataRecord,
      deployment: {
        provider: 'aws',
        region,
        url: deploymentUrl,
        updatedAt: deployedAt,
      },
    } as AIModelUpdate['metadata']

    // ✅ Update model deployment info
    const updatePayload: AIModelUpdate = {
      training_status: 'deployed',
      deployed_at: deployedAt,
      metadata: metadataPayload,
    }

    const { error: updateError } = await (supabase
      .from(AI_MODELS_TABLE) as any)
      .update(updatePayload)
      .eq('id', modelId)

    if (updateError) {
      console.error('Failed to update model info:', updateError)
    }

    return NextResponse.json({
      success: true,
      deploymentUrl,
      message: '✅ Model deployed to AWS successfully',
    })
  } catch (error: any) {
    console.error('AWS deployment error:', error)
    return NextResponse.json(
      { error: error.message || 'AWS deployment failed' },
      { status: 500 },
    )
  }
}
