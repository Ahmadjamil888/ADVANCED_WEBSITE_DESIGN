import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { 
  generateModelCode,
  analyzePrompt,
  findDataset,
  trainAIModel,
  deployToE2B,
  handleFollowUpConversation
} from "../../../inngest/functions";

// Create an API that serves all AI functions with correct IDs
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    // Core AI workspace functions with matching IDs
    generateModelCode,           // zehanx-ai-workspace-generate-model-code
    analyzePrompt,              // zehanx-ai-workspace-analyze-prompt  
    findDataset,                // zehanx-ai-workspace-find-dataset
    trainAIModel,               // zehanx-ai-workspace-train-model
    deployToE2B,                // zehanx-ai-workspace-deploy-e2b
    handleFollowUpConversation, // zehanx-ai-workspace-follow-up
  ],
});