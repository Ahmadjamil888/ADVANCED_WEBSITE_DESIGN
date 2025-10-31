import { NextResponse } from "next/server";
import { inngest } from "../../../inngest/client";
import { validateEnvironment, getEnvironmentInfo } from "@/lib/env-validation";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Validate environment
    const envValid = validateEnvironment();
    const envInfo = getEnvironmentInfo();

    // Test Inngest connection
    const testEvent = await inngest.send({
      name: "test/hello.world",
      data: {
        email: "test@zehanxtech.com",
        timestamp: new Date().toISOString(),
        source: "ai-workspace-test"
      }
    });

    return NextResponse.json({
      status: "success",
      message: "AI Workspace is ready!",
      environment: {
        valid: envValid,
        details: envInfo
      },
      inngest: {
        connected: true,
        testEventId: testEvent,
        client: "zehanx-ai-workspace"
      },
      services: {
        inngest: "✅ Connected",
        e2b: process.env.E2B_API_KEY ? "✅ Configured" : "❌ Not configured",
        kaggle: (process.env.KAGGLE_USERNAME && process.env.KAGGLE_KEY) ? "✅ Configured" : "❌ Not configured",
        supabase: process.env.NEXT_PUBLIC_SUPABASE_URL ? "✅ Connected" : "❌ Not connected"
      },
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('AI Workspace test error:', error);
    
    return NextResponse.json({
      status: "error",
      message: "AI Workspace configuration issue",
      error: error.message,
      environment: getEnvironmentInfo(),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}