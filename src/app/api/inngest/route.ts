import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { generateCompleteAIModel, deployToHuggingFace } from "../../../inngest/functions";

// Create an API that serves all AI functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    generateCompleteAIModel, // Complete AI model generation function
    deployToHuggingFace, // Hugging Face deployment function
  ],
});