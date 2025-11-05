import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import {
  testFunction,
  generateModelCode,
  trainAIModel,
  findDataset,
  analyzePrompt,
  deployToE2B,
  handleFollowUpConversation,
} from "../../../inngest/functions";

console.log("🔧 Registering Inngest functions...");

const functions = [
  testFunction,                // test-function
  generateModelCode,           // zehanx-ai-workspace-generate-model-code
  analyzePrompt,               // zehanx-ai-workspace-analyze-prompt
  findDataset,                 // find-dataset
  trainAIModel,                // train-ai-model
  deployToE2B,                 // zehanx-ai-workspace-deploy-e2b
  handleFollowUpConversation,  // zehanx-ai-workspace-follow-up
];

console.log("📋 Functions to register:", functions.map(f => f.id || f.name));

// Create an API that serves all AI functions with correct IDs
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions,
  signingKey: process.env.INNGEST_SIGNING_KEY,
});