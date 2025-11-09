import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
import {
  testFunction,
  generateModelCode,
  trainAIModel,
  deployToE2B,
} from "../../../inngest/functions";

console.log("🔧 Registering Inngest functions...");

// Only include actual Inngest functions (not helper functions like analyzePrompt and findDataset)
const functions = [
  testFunction,                // test-function
  generateModelCode,           // ai-workspace-generate-model
  trainAIModel,                // ai-train-model
  deployToE2B,                 // ai-deploy-e2b
];

console.log("📋 Functions to register:", functions.map(f => f.id || f.name));

// Create an API that serves all AI functions with correct IDs
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions,
  signingKey: process.env.INNGEST_SIGNING_KEY,
});