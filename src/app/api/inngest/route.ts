import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { generateCompleteAIModel, deployToHuggingFace } from "../../../inngest/functions";
import { 
  helloWorld, 
  generateModelCode, 
  trainAIModel, 
  deployToHuggingFace as deployHF, 
  findDataset 
} from "../../../lib/inngest/functions";

// Create an API that serves all AI functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    // Main AI functions
    generateCompleteAIModel, // Complete AI model generation function
    deployToHuggingFace, // Hugging Face deployment function
    
    // Legacy functions for compatibility
    helloWorld,
    generateModelCode,
    trainAIModel,
    deployHF,
    findDataset,
  ],
});